from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from ..core import (
    Ball, EventFlags, Player, Team, WorldState,
)
from .active import ActiveMarkerDetector
from .ball import BallManager
from .models import RTDETRAdapter
from .player import PlayerDetector
from .score import ScoreDetector
from .team import TeamClassifier
from .tracker import PlayerTracker
from .ui_mask import UIMask
from ..world.world import WorldModel
from ..world.temporal import WorldHistory
from ..world.game_mode import GameModeEstimator


class PerceptionEngine:
    """SCREEN -> detections -> tracking -> canonical Football World State."""

    def __init__(self, settings):
        self.settings = settings
        self.ui = UIMask(settings.ui)
        self.model = RTDETRAdapter(settings)
        self.player_detector = PlayerDetector(settings.player, self.ui)
        self.team = TeamClassifier(settings.player)
        self.active = ActiveMarkerDetector(settings.active)
        self.ball = BallManager(settings.ball, self.ui)
        self.tracker = PlayerTracker(settings.tracking)
        self.score = ScoreDetector(settings.score, async_mode=True)
        self.world = WorldModel(settings.world, settings.tactical)
        self.history = WorldHistory(settings.world.temporal_history, settings.world.world_temporal_ball_max_jump)
        self.game_mode = GameModeEstimator()
        self._last_game_mode = "UNKNOWN"
        self.last_stamp = None
        self.frame_id = 0
        self.prev_possession = Team.UNKNOWN
        self.prev_owner_id = -1
        self.last_score = (0, 0)
        self.last_score_initialized = False
        self._goal_candidate = None
        self._goal_candidate_frame = -1
        self._last_shot_frame = -10_000
        self._goal_latch_until = -1
        self._previous_ball = None
        self._fast_frame_counter = 0
        self._active_memory_id = -1
        self._active_memory_time = 0.0
        self._active_memory_conf = 0.0

    def close(self):
        self.score.close()
        self.model.close()

    def reset(self):
        self.active.reset()
        self.ball.reset()
        self.tracker.reset()
        self.score.reset()
        self.world.reset()
        self.history.reset()
        self._last_game_mode = "UNKNOWN"
        self.last_stamp = None
        self.frame_id = 0
        self.prev_possession = Team.UNKNOWN
        self.prev_owner_id = -1
        self.last_score = (0, 0)
        self.last_score_initialized = False
        self._goal_candidate = None
        self._goal_candidate_frame = -1
        self._last_shot_frame = -10_000
        self._goal_latch_until = -1
        self._previous_ball = None
        self._fast_frame_counter = 0
        self._active_memory_id = -1
        self._active_memory_time = 0.0
        self._active_memory_conf = 0.0

    @staticmethod
    def _score_goal_flags(previous_score, current_score, changed):
        changed = bool(changed)
        if not changed:
            return False, False
        return bool(current_score[0] > previous_score[0]), bool(current_score[1] > previous_score[1])

    def _goal_latch_open(self):
        return self.frame_id > self._goal_latch_until

    @staticmethod
    def _det_to_player(det, team_name, team_conf, appearance):
        return replace(
            det,
            team=team_name,
            team_confidence=team_conf,
            appearance=tuple(np.asarray(appearance, np.float32).tolist()),
        )

    def _merge_players(self, validated, recovered):
        merged = list(validated)
        # CV recovery may fill holes, but it is explicitly secondary to both the
        # current detector set and persistent track identities. This prevents
        # white/dark field components from inflating the roster every other frame.
        recovery_added = 0
        existing_tracks = list(self.tracker.tracks.values())
        for candidate in recovered:
            if recovery_added >= 2:
                break
            cp = np.asarray(candidate.center, np.float32)
            nearest_det = min(
                (float(np.linalg.norm(cp - np.asarray(det.center, np.float32))), det)
                for det in validated
            ) if validated else None
            if nearest_det is not None and nearest_det[0] <= 0.040:
                continue
            nearest_track = min(
                (float(np.linalg.norm(cp - np.asarray(track.predict(1/30), np.float32))), track)
                for track in existing_tracks
                if track.lifecycle.value != "DEAD"
            ) if existing_tracks else None
            if nearest_track is not None and nearest_track[0] <= 0.060:
                continue
            merged.append(candidate)
            recovery_added += 1
        return merged[: int(self.settings.player.max_players)]

    def step(self, frame, timestamp=None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        step_started = time.perf_counter()
        stage_t = step_started
        self.frame_id += 1
        dt = 1.0 / max(1.0, float(self.settings.runtime.perception_hz))
        if self.last_stamp is not None:
            dt = float(np.clip(timestamp - self.last_stamp, 1e-3, 0.20))
        self.last_stamp = timestamp

        # Submit the current frame without ever waiting for RT-DETR. The
        # worker publishes source timestamp + completion metadata, so freshness
        # is based on how old the detector measurement actually is.
        self.model.submit(frame, timestamp=timestamp, frame_id=self.frame_id)
        model = self.model.read()
        t_model = time.perf_counter()
        model_age_s = (
            float(timestamp - model.source_timestamp)
            if model.source_timestamp > 0.0
            else float("inf")
        )
        model_fresh = model.source_timestamp > 0.0 and 0.0 <= model_age_s <= self.settings.player.max_model_stale_s
        model_recent = model.source_timestamp > 0.0 and 0.0 <= model_age_s <= self.settings.player.model_reuse_stale_s
        # RT-DETR is a slow periodic re-detector on CPU. Its player proposals
        # remain useful to the tracker for a bounded reuse window; freshness is
        # still reported separately and is never faked. Ball proposals are much
        # more timing-sensitive and are consumed only while fresh.
        model_persons = list(model.persons) if model_recent else []
        if model_persons and not model_fresh:
            # The CPU RT-DETR result can be useful for identity refresh while it
            # is too old to be treated as a current measurement. Mark it stale
            # so the tracker blends it with current-frame optical flow instead
            # of snapping players back to the old detector geometry.
            model_persons = [replace(d, source="RTDETR_STALE") for d in model_persons]
        model_balls = list(model.balls) if model_fresh else []

        validated, pmetrics = self.player_detector.validate_model(frame, model_persons)
        recovered = []
        self._fast_frame_counter += 1
        # CV recovery is expensive and supplements, rather than replaces, the
        # persistent tracker. Run it every second frame unless the global detector
        # has gone stale, in which case run every frame to refill missing tracks.
        run_recovery = (
            self.settings.player.enable_cv_recovery
            and (model_recent or (self._fast_frame_counter % 2 == 0))
        )
        if run_recovery:
            recovered = self.player_detector.cv_recovery(frame)
        detections = self._merge_players(validated, recovered)
        t_players = time.perf_counter()

        classified = []
        for det in detections:
            team, conf, appearance = self.team.classify(frame, det)
            classified.append(
                self._det_to_player(
                    det, team, conf, appearance
                )
            )

        tracks = self.tracker.update(
            classified,
            dt,
            timestamp,
            frame=frame,
        )
        t_tracking = time.perf_counter()

        # Active marker and association.
        marker, marker_candidates = self.active.detect(frame)
        active_id, active_conf = self.active.associate(marker, tracks)
        # A fragmented/very thin selector may evade global connected-components.
        # Recover only inside bounded player-head windows; this is still visual
        # evidence and does not invent an active player from distance alone.
        if active_id < 0 or active_conf < self.settings.active.min_association_conf:
            recovered_marker, recovered_candidates = self.active.recover_from_tracks(frame, tracks)
            if recovered_marker is not None:
                recovered_id, recovered_conf = self.active.associate(recovered_marker, tracks)
                if recovered_id >= 0 and recovered_conf > active_conf:
                    marker = recovered_marker
                    marker_candidates = marker_candidates + recovered_candidates
                    active_id, active_conf = recovered_id, recovered_conf
        # Maintain a bounded semantic memory of the last directly observed active player.
        # This is not reported as raw active_valid; it is used only for the control contract.
        active_track = next((t for t in tracks if t.track_id == active_id), None)
        raw_active_valid = bool(
            active_id >= 0
            and active_conf >= self.settings.active.min_association_conf
            and active_track is not None
        )
        if raw_active_valid:
            self._active_memory_id = int(active_id)
            self._active_memory_time = float(timestamp)
            self._active_memory_conf = float(active_conf)
        else:
            age = float(timestamp - self._active_memory_time) if self._active_memory_time > 0 else float("inf")
            if age > self.settings.player.control_active_grace_s:
                self._active_memory_id = -1
                self._active_memory_time = 0.0
                self._active_memory_conf = 0.0
        t_active = time.perf_counter()

        # Ball fusion.
        ball = self.ball.update(frame, model_balls, players=tracks, active_track_id=active_id)
        t_ball = time.perf_counter()

        # Convert tracks to semantic players; UNKNOWN is retained.
        def as_player(track):
            team = {
                "OUR": Team.OUR,
                "OPP": Team.OPP,
                "UNKNOWN": Team.UNKNOWN,
            }.get(track.team, Team.UNKNOWN)
            return Player(
                track_id=track.track_id,
                x=float(track.pos[0]),
                y=float(track.pos[1]),
                vx=float(track.vel[0]),
                vy=float(track.vel[1]),
                team=team,
                team_confidence=float(track.team_conf),
                confidence=float(track.confidence),
                bbox=track.bbox,
                role=track.role,
                missed=track.missed,
            )

        players_screen = [as_player(t) for t in tracks]
        # Exactly one screen->canonical attacking-coordinate transform lives here.
        players = [self.world.canonical_player(p) for p in players_screen]
        ball_world = self.world.canonical_ball(ball)
        # Short LOST gaps are still part of the maintained world when the
        # tracker is carrying them with current-frame flow/prediction. This is
        # bounded by TrackingConfig.max_missed; it is not an unlimited ghost.
        own = [p for p in players if p.team == Team.OUR and p.missed <= 8]
        opp = [p for p in players if p.team == Team.OPP and p.missed <= 8]
        unknown = [p for p in players if p.team == Team.UNKNOWN and p.missed <= 8]
        active_player = next((p for p in players if p.track_id == active_id), None)

        all_players = own + opp + unknown
        designated_player, designated_distance, designated_conf = self.world.possession.designated(all_players, ball_world, dt=dt)
        designated_id = int(designated_player.track_id) if designated_player is not None else -1

        # World model derives possession/tactical state from canonical players + ball.
        possession, tactical, predictions = self.world.update(
            own,
            opp,
            ball_world,
            dt,
        )
        t_world = time.perf_counter()

        # Score OCR. score values are canonical: controlled side is OUR.
        if self.settings.score.controlled_side not in {"left", "right"}:
            raise ValueError("score.controlled_side must be 'left' or 'right'")
        score_info = self.score.update(frame, now=timestamp)
        t_score = time.perf_counter()
        if self.settings.score.controlled_side == "right":
            score_for = int(score_info["right"])
            score_against = int(score_info["left"])
        else:
            score_for = int(score_info["left"])
            score_against = int(score_info["right"])

        previous_score = tuple(score_info.get("previous_score", (score_for, score_against)))
        score_changed = bool(score_info.get("score_changed", False))
        score_observed_left = score_info.get("observed_left")
        score_observed_right = score_info.get("observed_right")

        score_goal_for, score_goal_against = self._score_goal_flags(
            previous_score, (score_for, score_against), score_changed
        )
        events = EventFlags(
            score_changed=score_changed,
            goal_for=score_goal_for,
            goal_against=score_goal_against,
        )
        if score_goal_for or score_goal_against:
            self._goal_latch_until = self.frame_id + 20

        # ----------------------------------------------------
        # Goal fallback from ball trajectory.
        # Score OCR is preferred, but gameplay can hide/blur the HUD.
        # Only crossing the goal line inside the goal mouth can promote
        # the fallback; a random near-goal ball is not enough.
        # ----------------------------------------------------
        previous_ball = self._previous_ball
        if ball_world.valid and not ball_world.predicted and previous_ball is not None:
            px, py = previous_ball
            crossed_for = (
                px < self.settings.ball.goal_cross_x_for
                <= ball_world.x
                and self.settings.ball.goal_y_min <= ball_world.y <= self.settings.ball.goal_y_max
                and ball_world.vx >= self.settings.ball.goal_min_vx
            )
            crossed_against = (
                px > self.settings.ball.goal_cross_x_against
                >= ball_world.x
                and self.settings.ball.goal_y_min <= ball_world.y <= self.settings.ball.goal_y_max
                and ball_world.vx <= -self.settings.ball.goal_min_vx
            )
            if crossed_for:
                self._goal_candidate = "for"
                self._goal_candidate_frame = self.frame_id
            elif crossed_against:
                self._goal_candidate = "against"
                self._goal_candidate_frame = self.frame_id

        # If the ball disappears immediately after crossing the line, promote
        # the candidate to a goal event.
        if (
            self._goal_latch_open()
            and self._goal_candidate == "for"
            and self.frame_id - self._goal_candidate_frame <= self.settings.ball.goal_candidate_timeout_frames
            and not events.goal_for and not events.goal_against
            and (not ball_world.valid or ball_world.predicted)
        ):
            events.goal_for = True
            self._goal_latch_until = self.frame_id + 20
        elif (
            self._goal_latch_open()
            and self._goal_candidate == "against"
            and self.frame_id - self._goal_candidate_frame <= self.settings.ball.goal_candidate_timeout_frames
            and not events.goal_for and not events.goal_against
            and (not ball_world.valid or ball_world.predicted)
        ):
            events.goal_against = True
            self._goal_latch_until = self.frame_id + 20

        if events.goal_for or events.goal_against:
            self._goal_candidate = None
            self._goal_candidate_frame = -1

        self._previous_ball = (ball_world.x, ball_world.y) if ball_world.valid and not ball_world.predicted else previous_ball

        # Possession transitions.
        if possession.owner_team == Team.OUR and self.prev_possession != Team.OUR:
            events.ball_won = True
        if possession.owner_team == Team.OPP and self.prev_possession == Team.OUR:
            events.ball_lost = True
        self.prev_possession = possession.owner_team
        self.prev_owner_id = possession.owner_track_id

        # Conservative shot-on-target heuristic; never used as goal substitute.
        if (
            possession.owner_team == Team.OUR
            and ball_world.valid
            and ball_world.x >= self.settings.tactical.shot_x
            and ball_world.vx >= 0.20
            and tactical.shot_quality >= 0.55
            and self.frame_id - self._last_shot_frame >= 20
        ):
            events.shot_on_target = True
            self._last_shot_frame = self.frame_id

        # Match done is deliberately false unless a future game-mode detector confirms it.
        events.match_done = False

        # World validity is now temporal and does not require the cyan active
        # marker or a named possession owner on every frame. This mirrors the
        # separation between active/designated/possession semantics used by
        # structured football RL environments.
        trusted_ball = (
            ball_world.valid
            and ball_world.confidence >= self.settings.ball.accept_conf
            and ball_world.source in {"DETECTOR", "CV"}
            and not ball_world.predicted
        )
        confirmed_recent = [p for p in players if p.missed <= 8 and p.confidence >= 0.25]
        stable_players = len(confirmed_recent) >= 5
        players_consistent = len([p for p in players if p.missed <= 8]) >= 5
        designated_valid = designated_id >= 0 and designated_conf >= self.settings.world.designated_min_confidence
        possession_valid = bool(
            ball_world.valid
            and (
                possession.owner_track_id >= 0
                or (self.settings.world.allow_world_without_possession_owner and possession.confidence >= 0.34)
            )
        )

        candidate_world = bool(trusted_ball and stable_players and players_consistent and designated_valid and possession_valid)

        game_mode, game_mode_conf = self.game_mode.classify(
            ball_valid=bool(ball_world.valid and not ball_world.predicted),
            player_count=len(confirmed_recent),
            score_changed=bool(score_changed),
            previous=self._last_game_mode,
        )
        self._last_game_mode = game_mode

        # Build a provisional state for temporal memory before final validity.
        temporal_world = WorldState(
            frame_id=self.frame_id, timestamp=timestamp, dt=dt, ball=ball_world,
            active_track_id=active_id, active_valid=(active_id >= 0 and active_conf >= self.settings.active.min_association_conf),
            designated_track_id=designated_id, designated_confidence=float(designated_conf),
            possession=possession, world_valid=candidate_world,
        )
        self.history.push(temporal_world)
        temporal = self.history.summary()
        temporal_valid = bool(
            temporal["depth"] >= self.settings.world.world_temporal_min_depth
            and temporal["continuity_valid"]
        )
        world_valid = bool(candidate_world and (temporal_valid or temporal["depth"] < self.settings.world.world_temporal_min_depth + 1))

        # A short predicted-ball bridge is acceptable for WorldState/control when
        # the last real measurement is recent and temporal continuity is intact.
        # UNKNOWN remains fail-closed.
        predicted_bridge = bool(
            self.settings.world.allow_world_with_short_ball_prediction
            and ball_world.predicted
            and ball_world.valid
            and ball_world.confidence >= self.settings.world.short_ball_prediction_min_conf
            and float(self.ball.last_debug.get("prediction_age_s") or 999.0) <= self.settings.world.short_ball_prediction_max_s
            and temporal_valid
        )
        if predicted_bridge:
            world_valid = bool(
                stable_players
                and players_consistent
                and designated_valid
                and possession_valid
                and (temporal_valid or temporal["depth"] < self.settings.world.world_temporal_min_depth + 1)
            )

        memory_age = float(timestamp - self._active_memory_time) if self._active_memory_time > 0 else float("inf")
        memory_track = next((p for p in players if p.track_id == self._active_memory_id), None)
        effective_active_id = int(active_id)
        effective_active_conf = float(active_conf)
        control_memory = False
        if not raw_active_valid and memory_track is not None and memory_age <= self.settings.player.control_active_grace_s:
            if memory_track.missed <= 8 and memory_track.team in {Team.OUR, Team.UNKNOWN} and memory_track.confidence >= self.settings.player.control_active_min_conf:
                effective_active_id = int(memory_track.track_id)
                effective_active_conf = float(max(self.settings.player.control_active_min_conf, self._active_memory_conf * np.exp(-memory_age / max(self.settings.player.control_active_grace_s, 1e-3))))
                control_memory = True

        control_player = next((p for p in players if p.track_id == effective_active_id), None)

        model_control_ok = (
            not self.settings.player.require_model_for_control
            or (model.source_timestamp > 0.0 and model_age_s <= self.settings.player.control_model_stale_s)
        )
        # Control remains strict: an actual active player is required, but a
        # short active-marker weakness no longer destroys WorldState itself.
        control_valid = bool(
            model_control_ok
            and world_valid
            and effective_active_id >= 0
            and effective_active_conf >= self.settings.player.control_active_min_conf
            and len(confirmed_recent) >= 5
            and game_mode == "PLAY"
            and (not ball_world.predicted or predicted_bridge)
        )

        predictions_out = {}
        for horizon in self.settings.world.prediction_horizons:
            key = f"{horizon:.2f}"
            if key in predictions:
                predictions_out[f"ball:{key}"] = predictions[key]
        predictions_out.update(
            self.world.predictions_for_players(
                own[:11] + opp[:11]
            )
        )

        perception = {
            "raw_person": float(pmetrics["raw_person"]),
            "pitch_validated": float(pmetrics["pitch_validated"]),
            "ui_rejected": float(pmetrics["ui_rejected"]),
            "size_rejected": float(pmetrics["size_rejected"]),
            "pitch_rejected": float(pmetrics["pitch_rejected"]),
            "player_candidates": float(len(detections)),
            "tracks": float(len(tracks)),
            "stable_player_count": float(len(confirmed_recent)),
            "stable_players": float(stable_players),
            "flow_tracks": float(getattr(self.tracker, "last_flow_tracks", 0)),
            "flow_usable": float(getattr(self.tracker, "last_flow_count", 0)),
            "marker_candidates": float(len(marker_candidates)),
            "active_confidence": float(active_conf),
            "control_track_id": float(effective_active_id),
            "control_track_confidence": float(effective_active_conf),
            "control_active_memory": float(control_memory),
            "prediction_bridge": float(predicted_bridge),
            "designated_track_id": float(designated_id),
            "designated_confidence": float(designated_conf),
            "designated_distance": float(designated_distance),
            "temporal_depth": float(temporal["depth"]),
            "temporal_ball_dx": float(temporal["ball_dx"]),
            "temporal_ball_dy": float(temporal["ball_dy"]),
            "temporal_active_streak": float(temporal["active_streak"]),
            "temporal_possession_streak": float(temporal["possession_streak"]),
            "temporal_valid": float(temporal_valid),
            "game_mode_confidence": float(game_mode_conf),
            "ball_confidence": float(ball.confidence),
            "ball_predicted": float(ball.predicted),
            "score_confidence": float(score_info.get("confidence", 0.0)),
            "score_initialized": float(bool(score_info.get("initialized", False))),
            "score_changed": float(bool(score_info.get("score_changed", False))),
            "score_invalid_transition": float(bool(score_info.get("invalid_transition", False))),
            "score_invalid_transition_count": float(getattr(self.score, "invalid_transition_count", 0)),
            "score_observed_left": float(score_observed_left) if score_observed_left is not None else -1.0,
            "score_observed_right": float(score_observed_right) if score_observed_right is not None else -1.0,
            "model_age_s": None if not np.isfinite(model_age_s) else float(model_age_s),
            "model_recent": model_recent,
            "model_inference_latency_s": float(model.inference_latency_s),
            "model_source_frame_id": float(model.frame_id),
        }

        def det_json(items):
            return [
                {
                    "bbox": tuple(float(v) for v in d.bbox),
                    "center": tuple(float(v) for v in d.center),
                    "source": d.source,
                    "confidence": float(d.confidence),
                    "team": d.team,
                    "team_confidence": float(d.team_confidence),
                }
                for d in items
            ]

        debug = {
            "active_marker": marker,
            "active_track_id": active_id,
            "control_track_id": effective_active_id,
            "control_active_memory": bool(control_memory),
            "designated_track_id": designated_id,
            "designated_confidence": float(designated_conf),
            "game_mode": game_mode,
            "game_mode_confidence": float(game_mode_conf),
            "temporal": dict(temporal),
            "active_candidates": marker_candidates,
            "ball_source": ball.source,
            "score": (score_for, score_against),
            "score_initialized": bool(score_info.get("initialized", False)),
            "score_changed": events.score_changed,
            "model_fresh": model_fresh,
            "model_recent": model_recent,
            "model_age_s": None if not np.isfinite(model_age_s) else float(model_age_s),
            "model_recent": model_recent,
            "model_source_frame_id": int(model.frame_id),
            "model_inference_latency_s": float(model.inference_latency_s),
            "model_age_at_completion_s": float(model.age_at_completion_s),
            "model_device": self.model.device,
            "model_runtime": self.model.runtime_metrics(timestamp),
            "stages": {
                "raw_rtdetr_players": det_json(model_persons),
                "pitch_validated": det_json(validated),
                "recovery": det_json(recovered),
                "merged_players": det_json(classified),
                "marker": marker,
                "marker_candidates": marker_candidates,
                "tracks": [
                    {
                        "track_id": int(t.track_id),
                        "pos": tuple(float(v) for v in t.pos),
                        "bbox": t.bbox,
                        "team": t.team,
                        "confidence": float(t.confidence),
                        "missed": int(t.missed),
                        "lifecycle": str(t.lifecycle),
                    }
                    for t in tracks
                ],
                "ball": {
                    "x": float(ball.x),
                    "y": float(ball.y),
                    "world_x": float(ball_world.x),
                    "world_y": float(ball_world.y),
                    "confidence": float(ball.confidence),
                    "source": ball.source,
                    "predicted": bool(ball.predicted),
                    "valid": bool(ball.valid),
                    **self.ball.last_debug,
                },
            },
        }

        debug["score_previous"] = list(previous_score)
        debug["score_observed_left"] = score_observed_left
        debug["score_observed_right"] = score_observed_right
        debug["score_observed_left_confidence"] = float(score_info.get("observed_left_confidence", 0.0))
        debug["score_observed_right_confidence"] = float(score_info.get("observed_right_confidence", 0.0))
        debug["timings"] = {
            "model": t_model - stage_t,
            "players": t_players - t_model,
            "tracking": t_tracking - t_players,
            "active": t_active - t_tracking,
            "ball": t_ball - t_active,
            "world": t_world - t_ball,
            "score": t_score - t_world,
            "total": time.perf_counter() - step_started,
        }
        debug["score_debug"] = dict(self.score.last_debug)
        debug["score_invalid_transition_count"] = int(getattr(self.score, "invalid_transition_count", 0))

        return WorldState(
            frame_id=self.frame_id,
            timestamp=timestamp,
            dt=dt,
            ball=ball_world,
            own_players=own[:11],
            opp_players=opp[:11],
            unknown_players=unknown,
            active_track_id=active_id,
            active_player=active_player,
            active_confidence=float(active_conf),
            control_player=control_player,
            control_track_id=effective_active_id,
            control_track_confidence=float(effective_active_conf),
            control_active_memory=bool(control_memory),
            designated_track_id=designated_id,
            designated_player=designated_player,
            designated_confidence=float(designated_conf),
            possession=possession,
            tactical=tactical,
            events=events,
            score_for=score_for,
            score_against=score_against,
            score_confidence=float(score_info.get("confidence", 0.0)),
            clock_s=float(score_info.get("clock_s", 0.0)),
            game_mode=game_mode,
            game_mode_confidence=float(game_mode_conf),
            world_valid=world_valid,
            control_valid=control_valid,
            players_valid=len(players) >= 5,
            ball_valid=ball.valid,
            active_valid=(active_id >= 0 and active_conf >= self.settings.active.min_association_conf),
            possession_valid=possession_valid,
            temporal_valid=temporal_valid,
            temporal_depth=int(temporal["depth"]),
            temporal_ball_dx=float(temporal["ball_dx"]),
            temporal_ball_dy=float(temporal["ball_dy"]),
            temporal_active_streak=int(temporal["active_streak"]),
            temporal_possession_streak=int(temporal["possession_streak"]),
            perception=perception,
            predictions=predictions_out,
            debug=debug,
        )

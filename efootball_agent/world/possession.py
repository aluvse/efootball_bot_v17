from __future__ import annotations

import numpy as np

from ..core import Possession, Team


class PossessionEstimator:
    """Ball association + sticky possession/designated-player state.

    Designated player is a semantic ball-associated player, not the cyan active
    marker and not necessarily the current possession owner. Both layers retain
    bounded temporal memory so a single weak visual frame does not cause a
    wholesale identity flip.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = Possession()
        self.designated_track_id = -1
        self.designated_confidence = 0.0
        self.designated_age_s = 0.0
        self.designated_team = Team.UNKNOWN
        self.designated_point: np.ndarray | None = None
        self.last_ball_point: np.ndarray | None = None
        self.last_ball_time = 0.0
        self.weak_ball_age_s = 0.0

    def reset(self):
        self.state = Possession()
        self.designated_track_id = -1
        self.designated_confidence = 0.0
        self.designated_age_s = 0.0
        self.designated_team = Team.UNKNOWN
        self.designated_point = None
        self.last_ball_point = None
        self.last_ball_time = 0.0
        self.weak_ball_age_s = 0.0

    @staticmethod
    def _player_point(player) -> np.ndarray:
        foot_y = float(player.y)
        bbox = getattr(player, "bbox", None)
        if bbox is not None:
            _, y1, _, y2 = [float(v) for v in bbox]
            foot_y = float(np.clip(player.y + 0.50 * (y2 - y1), 0.0, 1.0))
        return np.array([float(player.x), foot_y], np.float32)

    def _candidate_score(self, player, ball):
        if ball is None:
            return 0.0, 1.0
        p = self._player_point(player)
        b = np.array([ball.x, ball.y], np.float32)
        distance = float(np.linalg.norm(b - p))
        rel_velocity = float(
            np.linalg.norm(
                np.array([ball.vx - player.vx, ball.vy - player.vy], np.float32)
            )
        )
        ball_speed = float(np.linalg.norm(np.array([ball.vx, ball.vy], np.float32)))
        distance_score = float(np.exp(-distance / max(self.cfg.possession_scale, 1e-4)))
        velocity_score = float(np.exp(-rel_velocity / 1.5))
        slow_ball = float(np.clip(1.0 - ball_speed / 2.5, 0.0, 1.0))
        control_score = (
            0.78 * distance_score
            + 0.14 * velocity_score
            + 0.08 * slow_ball
        )
        return float(np.clip(control_score, 0.0, 1.0)), distance

    def _designated_candidate_score(self, player, ball):
        base, distance = self._candidate_score(player, ball)
        score = 0.72 * base

        # Designated-player continuity: staying on the current ball-associated
        # player wins unless a new candidate is meaningfully better.
        if player.track_id == self.designated_track_id:
            score += float(getattr(self.cfg, "designated_previous_bonus", 0.12))

        # Stable team identity is a mild tie-breaker, not a hard filter; UNKNOWN
        # players remain eligible because existence and team semantics are split.
        if self.designated_team != Team.UNKNOWN and player.team == self.designated_team:
            score += float(getattr(self.cfg, "designated_team_bonus", 0.04))

        # A player moving consistently toward the ball receives a small bonus.
        ball_xy = np.array([ball.x, ball.y], np.float32)
        to_ball = ball_xy - self._player_point(player)
        norm = float(np.linalg.norm(to_ball))
        if norm > 1e-5:
            velocity = np.array([player.vx, player.vy], np.float32)
            velocity_norm = float(np.linalg.norm(velocity))
            if velocity_norm > 1e-4:
                alignment = float(np.dot(velocity, to_ball) / (velocity_norm * norm))
                score += 0.06 * max(0.0, alignment)

        return float(np.clip(score, 0.0, 1.0)), distance

    def best(self, players, ball):
        if not players or ball is None or not ball.valid:
            return None, 1.0, 0.0
        ranked = []
        for player in players:
            score, distance = self._candidate_score(player, ball)
            ranked.append((score, distance, player))
        score, distance, player = max(
            ranked,
            key=lambda x: (x[0], -x[1], -x[2].track_id),
        )
        return player, distance, score

    def designated(self, players, ball, dt: float = 0.0):
        """Temporally sticky designated-player association.

        The current candidate must beat the incumbent by a configurable margin
        before switching. Short ball gaps retain the previous designated player
        with a bounded confidence decay.
        """
        dt = max(0.0, float(dt))
        ball_valid = bool(ball is not None and ball.valid)
        if ball_valid:
            self.last_ball_point = np.array([ball.x, ball.y], np.float32)
            self.last_ball_time += dt
        else:
            self.last_ball_time += dt

        if not players:
            self.designated_confidence *= float(getattr(self.cfg, "designated_decay", 0.92))
            self.designated_age_s += dt
            return None, 1.0, self.designated_confidence

        candidate = None
        candidate_distance = 1.0
        candidate_score = 0.0
        if ball_valid:
            ranked = []
            for player in players:
                score, distance = self._designated_candidate_score(player, ball)
                ranked.append((score, distance, player))
            candidate_score, candidate_distance, candidate = max(
                ranked,
                key=lambda x: (x[0], -x[1], -x[2].track_id),
            )

        incumbent = next(
            (p for p in players if p.track_id == self.designated_track_id),
            None,
        )

        switch_margin = float(getattr(self.cfg, "designated_switch_margin", 0.10))
        on_conf = float(getattr(self.cfg, "designated_min_confidence", 0.45))
        hold_s = float(getattr(self.cfg, "designated_hold_s", 0.55))

        if candidate is not None:
            if incumbent is None:
                choose = candidate_score >= on_conf
            else:
                incumbent_score, _ = self._designated_candidate_score(incumbent, ball)
                choose = (
                    candidate.track_id == incumbent.track_id
                    or candidate_score >= incumbent_score + switch_margin
                    or self.designated_confidence < 0.25
                )
            if choose:
                switched = candidate.track_id != self.designated_track_id
                self.designated_track_id = int(candidate.track_id)
                self.designated_team = Team(candidate.team)
                self.designated_point = self._player_point(candidate)
                self.designated_confidence = float(
                    np.clip(
                        0.68 * self.designated_confidence + 0.32 * candidate_score
                        if not switched else candidate_score,
                        0.0,
                        0.99,
                    )
                )
                self.designated_age_s = 0.0 if switched else self.designated_age_s + dt
                return candidate, candidate_distance, self.designated_confidence

        # Ball weak/lost or the challenger did not clear the hysteresis margin:
        # keep the incumbent when it remains visible, otherwise retain identity
        # only for a short bounded grace interval.
        if incumbent is not None:
            incumbent_score, incumbent_distance = self._designated_candidate_score(
                incumbent,
                ball if ball_valid else type("BallProxy", (), {
                    "x": float(self.last_ball_point[0]) if self.last_ball_point is not None else 0.5,
                    "y": float(self.last_ball_point[1]) if self.last_ball_point is not None else 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "valid": True,
                })(),
            )
            self.designated_confidence = float(
                np.clip(
                    max(incumbent_score * 0.90, self.designated_confidence * 0.94),
                    0.0,
                    0.99,
                )
            )
            self.designated_age_s += dt
            return incumbent, incumbent_distance, self.designated_confidence

        if self.designated_track_id >= 0 and self.designated_age_s + dt <= hold_s:
            self.designated_confidence *= float(getattr(self.cfg, "designated_decay", 0.92))
            self.designated_age_s += dt
            return None, candidate_distance, self.designated_confidence

        self.designated_track_id = -1
        self.designated_team = Team.UNKNOWN
        self.designated_point = None
        self.designated_confidence = 0.0
        self.designated_age_s = 0.0
        return None, 1.0, 0.0

    def update(self, own, opp, ball, dt):
        dt = max(0.0, float(dt))
        if ball is None or not ball.valid:
            self.weak_ball_age_s += dt
            grace = float(getattr(self.cfg, "possession_unknown_grace_s", 0.30))
            if self.state.owner_team != Team.UNKNOWN and self.weak_ball_age_s <= grace:
                confidence = float(self.state.confidence * getattr(self.cfg, "possession_unknown_decay", 0.92))
                state_name = "OUR_RECEIVING" if self.state.owner_team == Team.OUR else "OPP_CONTROL"
                self.state = Possession(
                    state_name,
                    self.state.owner_team,
                    self.state.owner_track_id,
                    confidence,
                    self.state.age_s + dt,
                )
                return self.state
            self.state = Possession("LOOSE_BALL", Team.UNKNOWN, -1, 0.0, 0.0)
            return self.state

        self.weak_ball_age_s = 0.0
        own_p, own_d, own_score = self.best(own, ball)
        opp_p, opp_d, opp_score = self.best(opp, ball)

        # Possession continuity is asymmetric: keep OUR/OPP ownership through
        # brief visual ambiguity, but require a clear competing signal to switch.
        if self.state.owner_team == Team.OUR and own_p is not None and own_score >= self.cfg.possession_off:
            own_score = max(own_score, self.state.confidence * 0.94)
        if self.state.owner_team == Team.OPP and opp_p is not None and opp_score >= self.cfg.possession_off:
            opp_score = max(opp_score, self.state.confidence * 0.94)

        switch_margin = float(getattr(self.cfg, "possession_switch_margin", 0.10))
        if own_score >= self.cfg.possession_on and own_score > opp_score + switch_margin:
            target = Possession("OUR_CONTROL", Team.OUR, own_p.track_id, own_score, 0.0)
        elif opp_score >= self.cfg.possession_on and opp_score > own_score + switch_margin:
            target = Possession("OPP_CONTROL", Team.OPP, opp_p.track_id, opp_score, 0.0)
        elif own_score >= self.cfg.possession_off and own_score >= opp_score:
            owner_id = own_p.track_id if own_p is not None else self.state.owner_track_id
            target = Possession("OUR_RECEIVING", Team.OUR, owner_id, own_score, 0.0)
        elif opp_score >= self.cfg.possession_off and opp_score > own_score:
            target = Possession("CONTESTED", Team.UNKNOWN, -1, opp_score, 0.0)
        else:
            target = Possession("LOOSE_BALL", Team.UNKNOWN, -1, max(own_score, opp_score), 0.0)

        # Preserve prior owner through weak contests; sustained opponent evidence
        # can still replace it normally once the switch margin is cleared.
        if self.state.owner_team == Team.OUR and target.owner_team != Team.OUR:
            if own_score >= self.cfg.possession_off and own_score + switch_margin >= opp_score:
                target = Possession(
                    "OUR_RECEIVING",
                    Team.OUR,
                    self.state.owner_track_id,
                    max(self.state.confidence * 0.90, own_score),
                    self.state.age_s + dt,
                )
        elif self.state.owner_team == Team.OPP and target.owner_team != Team.OPP:
            if opp_score >= self.cfg.possession_off and opp_score + switch_margin >= own_score:
                target = Possession(
                    "OPP_CONTROL",
                    Team.OPP,
                    self.state.owner_track_id,
                    max(self.state.confidence * 0.90, opp_score),
                    self.state.age_s + dt,
                )

        if target.owner_team == self.state.owner_team and target.owner_track_id == self.state.owner_track_id:
            target.age_s = self.state.age_s + dt
        else:
            target.age_s = 0.0
        self.state = target
        return target

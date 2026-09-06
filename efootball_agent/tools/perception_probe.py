from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ..config import SETTINGS
from ..perception.pipeline import PerceptionEngine


def point_px(point, w, h):
    return int(point[0] * w), int(point[1] * h)


def draw_boxes(frame, items, label_mode="player"):
    out = frame.copy()
    h, w = out.shape[:2]
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        p1 = point_px((x1, y1), w, h)
        p2 = point_px((x2, y2), w, h)
        if label_mode == "team":
            text = f"{item.get('team','?')} {item.get('team_confidence',0):.2f}"
        else:
            text = f"{item.get('source','?')} {item.get('confidence',0):.2f}"
        cv2.rectangle(out, p1, p2, (0, 200, 255), 2)
        cv2.putText(out, text, (p1[0], max(15, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
    return out


def draw_tracks(frame, tracks):
    out = frame.copy()
    h, w = out.shape[:2]
    for track in tracks:
        x, y = point_px(track["pos"], w, h)
        cv2.circle(out, (x, y), 8, (255, 0, 255), 2)
        cv2.putText(out, f"T{track['track_id']} {track['team']} {track['lifecycle']}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), 1)
    return out


def draw_control(frame, world):
    out = frame.copy()
    ok = bool(world.control_valid)
    line = f"CONTROL={'VALID' if ok else 'BLOCKED'} world={world.world_valid} ball={world.ball.source}:{world.ball.confidence:.2f} active={world.active_confidence:.2f} poss={world.possession.confidence:.2f}"
    cv2.rectangle(out, (5, 5), (min(out.shape[1]-5, 900), 34), (0, 0, 0), -1)
    cv2.putText(out, line, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    if not ok:
        cv2.putText(out, "NO INPUT: safety neutral", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)
    return out


def draw_final(frame, world):
    out = frame.copy()
    h, w = out.shape[:2]
    for p in world.own_players:
        x, y = point_px((p.x, p.y), w, h)
        cv2.circle(out, (x, y), 8, (0, 180, 0), -1)
        cv2.putText(out, f"O{p.track_id}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)
    for p in world.opp_players:
        x, y = point_px((p.x, p.y), w, h)
        cv2.circle(out, (x, y), 8, (0, 0, 220), -1)
        cv2.putText(out, f"X{p.track_id}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 150), 1)
    for p in world.unknown_players:
        x, y = point_px((p.x, p.y), w, h)
        cv2.circle(out, (x, y), 7, (150, 150, 150), 1)
    if world.ball.valid or world.ball.predicted:
        x, y = point_px((world.ball.x, world.ball.y), w, h)
        cv2.circle(out, (x, y), 10, (0, 165, 255), 2)
        cv2.putText(out, f"BALL {world.ball.source}:{world.ball.confidence:.2f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 220), 1)
    if world.active_player is not None:
        x, y = point_px((world.active_player.x, world.active_player.y), w, h)
        cv2.circle(out, (x, y), 18, (255, 255, 0), 2)
    lines = [
        f"world_valid={world.world_valid} control_valid={world.control_valid}",
        f"players O/P/U={len(world.own_players)}/{len(world.opp_players)}/{len(world.unknown_players)}",
        f"active={world.active_track_id} conf={world.active_confidence:.2f}",
        f"possession={world.possession.state} conf={world.possession.confidence:.2f}",
        f"score={world.score_for}:{world.score_against} conf={world.score_confidence:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, line, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="V24.1 one-frame perception probe")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("probe_out"))
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)

    engine = PerceptionEngine(SETTINGS)
    try:
        t0 = time.perf_counter()
        submit_ts = time.monotonic()
        # Warm the asynchronous detector before running the complete pipeline.
        # Score OCR is deliberately isolated from this wait so model freshness
        # is measured against inference completion rather than OCR wall time.
        engine.model.submit(frame)

        # RT-DETR is intentionally asynchronous. Do not label the model
        # "stale" just because one hard-coded 0.4 s sleep expired while the
        # first CPU inference was still warming up. Poll the adapter for a
        # genuinely stamped result, bounded so the probe never hangs.
        deadline = time.perf_counter() + 8.0
        while time.perf_counter() < deadline:
            result = engine.model.read()
            if result.stamp >= submit_ts:
                break
            time.sleep(0.025)
        world = engine.step(frame, timestamp=time.monotonic())
        # Offline probes are single-frame by nature. Confirm only the HUD score
        # on the frozen image; do NOT rerun the entire perception pipeline,
        # otherwise ball/tracking state can advance into an artificial prediction.
        if SETTINGS.score.enabled and not bool(world.debug.get("score_initialized", False)):
            score_now = time.monotonic() + SETTINGS.score.interval_s + 0.01
            score_info = engine.score.update(frame, now=score_now, force_confirm=True)
            world.score_for = int(score_info.get("left", 0) if SETTINGS.score.controlled_side == "left" else score_info.get("right", 0))
            world.score_against = int(score_info.get("right", 0) if SETTINGS.score.controlled_side == "left" else score_info.get("left", 0))
            world.score_confidence = float(score_info.get("confidence", 0.0))
            world.clock_s = float(score_info.get("clock_s", world.clock_s))
            world.debug["score_initialized"] = bool(score_info.get("initialized", False))
            world.perception["score_initialized"] = float(bool(score_info.get("initialized", False)))
            world.perception["score_confidence"] = float(score_info.get("confidence", 0.0))
            world.perception["score_changed"] = float(bool(score_info.get("score_changed", False)))
            world.debug["score_observed_left"] = score_info.get("observed_left")
            world.debug["score_observed_right"] = score_info.get("observed_right")
            world.debug["score_observed_left_confidence"] = float(score_info.get("observed_left_confidence", 0.0))
            world.debug["score_observed_right_confidence"] = float(score_info.get("observed_right_confidence", 0.0))
        timings = dict(world.debug.get("timings", {}))
        timings["probe_wall"] = time.perf_counter() - t0

        stages = world.debug["stages"]
        cv2.imwrite(str(args.out / "00_original.jpg"), frame)
        cv2.imwrite(str(args.out / "01_raw_rtdetr.jpg"), draw_boxes(frame, stages["raw_rtdetr_players"]))
        cv2.imwrite(str(args.out / "02_pitch_validated.jpg"), draw_boxes(frame, stages["pitch_validated"]))
        cv2.imwrite(str(args.out / "03_team.jpg"), draw_boxes(frame, stages["merged_players"], "team"))

        active_img = frame.copy()
        marker = stages["marker"]
        if marker:
            x, y = point_px(marker, frame.shape[1], frame.shape[0])
            cv2.circle(active_img, (x, y), 18, (255, 255, 0), 2)
            cv2.putText(active_img, f"ACTIVE track={world.active_track_id} conf={world.active_confidence:.2f}", (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.imwrite(str(args.out / "04_active.jpg"), active_img)

        ball_img = frame.copy()
        bx = stages["ball"]
        x, y = point_px((bx["x"], bx["y"]), frame.shape[1], frame.shape[0])
        if bx["valid"] or bx["predicted"]:
            cv2.circle(ball_img, (x, y), 12, (0, 165, 255), 2)
        cv2.putText(ball_img, f"BALL {bx['source']} {bx['confidence']:.2f}", (10, frame.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(str(args.out / "05_ball.jpg"), ball_img)
        cv2.imwrite(str(args.out / "06_tracks.jpg"), draw_tracks(frame, stages["tracks"]))
        cv2.imwrite(str(args.out / "07_world.jpg"), draw_final(frame, world))
        cv2.imwrite(str(args.out / "08_control.jpg"), draw_control(frame, world))

        # Score debug: exact ROI crops side by side.
        left_crop = engine.score.crop(frame, SETTINGS.score.left_score_roi)
        right_crop = engine.score.crop(frame, SETTINGS.score.right_score_roi)
        score_debug = np.zeros((max(left_crop.shape[0], right_crop.shape[0]), left_crop.shape[1] + right_crop.shape[1], 3), np.uint8)
        score_debug[:left_crop.shape[0], :left_crop.shape[1]] = left_crop
        score_debug[:right_crop.shape[0], left_crop.shape[1]:] = right_crop
        score_debug = cv2.resize(score_debug, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(args.out / "score_debug.png"), score_debug)

        report = {
            "image": str(args.image),
            "capture": {"frame": [frame.shape[1], frame.shape[0]], "source": "offline_image", "window_only": True},
            "model": {
                "loaded": engine.model.detector is not None,
                "processor_loaded": engine.model.processor is not None,
                "fresh": bool(world.debug.get("model_fresh", False)),
                "source_age_s": float(max(0.0, world.timestamp - engine.model.read().stamp)) if engine.model.read().stamp else None,
                "load_error": None if engine.model.load_error is None else f"{type(engine.model.load_error).__name__}: {engine.model.load_error}",
            },
            "players": world.perception,
            "team": {
                "our": len(world.own_players),
                "opp": len(world.opp_players),
                "unknown": len(world.unknown_players),
            },
            "active": {
                "valid": world.active_valid,
                "track_id": world.active_track_id,
                "confidence": world.active_confidence,
                "marker_candidates": len(world.debug.get("active_candidates", [])),
                "matched_track_id": world.active_track_id,
                "association_score": float(max(0.0, 1.0 - world.active_confidence)),
            },
            "ball": {
                "valid": world.ball.valid,
                "confidence": world.ball.confidence,
                "source": world.ball.source,
                "predicted": world.ball.predicted,
                "position": [world.ball.x, world.ball.y],
                "velocity": [world.ball.vx, world.ball.vy],
                **{k: v for k, v in world.debug.get("stages", {}).get("ball", {}).items() if k in {
                    "rtdetr_confidence", "cv_confidence", "selected_source", "final_confidence",
                    "field_valid", "ui_rejected", "temporal_consistency", "cv_candidates", "detector_candidates",
                    "fusion_corroborated", "fusion_distance", "fusion_confidence_bonus", "near_foot",
                    "foot_distance", "footpoint_candidates"
                }},
            },
            "tracking": {
                "players": len(stages["tracks"]),
                "unknown": len(world.unknown_players),
                "tracks": stages["tracks"],
            },
            "possession": {
                "state": world.possession.state,
                "owner_track_id": world.possession.owner_track_id,
                "owner_team": int(world.possession.owner_team),
                "confidence": world.possession.confidence,
                "age_s": world.possession.age_s,
            },
            "score": {
                "left": int(world.score_for if SETTINGS.score.controlled_side == "left" else world.score_against),
                "right": int(world.score_for if SETTINGS.score.controlled_side == "right" else world.score_against),
                "score_for": world.score_for,
                "score_against": world.score_against,
                "score_confidence": world.score_confidence,
                "initialized": bool(world.debug.get("score_initialized", False)),
                "score_changed": bool(world.events.score_changed),
                "previous_score": list(world.debug.get("score_previous", [world.score_for, world.score_against])),
                "clock_s": world.clock_s,
                "observed_left": world.debug.get("score_observed_left"),
                "observed_right": world.debug.get("score_observed_right"),
                "observed_left_confidence": world.debug.get("score_observed_left_confidence", 0.0),
                "observed_right_confidence": world.debug.get("score_observed_right_confidence", 0.0),
            },
            "world": {
                "valid": world.world_valid,
                "ball_valid": world.ball_valid,
                "active_valid": world.active_valid,
                "players_valid": world.players_valid,
                "possession_valid": world.possession_valid,
                "game_mode": world.game_mode,
                "game_mode_confidence": world.game_mode_confidence,
            },
            "control": {
                "valid": world.control_valid,
                "model_fresh": bool(world.debug.get("model_fresh", False)),
            },
            "events": world.events.__dict__,
            "timings": timings,
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()

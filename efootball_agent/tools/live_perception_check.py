from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efootball_agent.config import SETTINGS
from efootball_agent.perception.capture import WindowCapture
from efootball_agent.perception.pipeline import PerceptionEngine


def _world_sample(world):
    p = world.perception or {}
    return {
        "frame_id": int(world.frame_id),
        "timestamp": float(world.timestamp),
        "ball": {
            "valid": bool(world.ball.valid),
            "source": str(world.ball.source),
            "predicted": bool(world.ball.predicted),
            "confidence": float(world.ball.confidence),
            "x": float(world.ball.x),
            "y": float(world.ball.y),
        },
        "active": {
            "valid": bool(world.active_valid),
            "track_id": int(world.active_track_id),
            "confidence": float(world.active_confidence),
        },
        "designated": {
            "track_id": int(world.designated_track_id),
            "confidence": float(world.designated_confidence),
        },
        "players": {
            "own": int(len(world.own_players)),
            "opp": int(len(world.opp_players)),
            "unknown": int(len(world.unknown_players)),
            "total": int(len(world.own_players) + len(world.opp_players) + len(world.unknown_players)),
            "raw_person": int(p.get("raw_person", 0)),
            "pitch_validated": int(p.get("pitch_validated", 0)),
            "ui_rejected": int(p.get("ui_rejected", 0)),
            "pitch_rejected": int(p.get("pitch_rejected", 0)),
        },
        "temporal": {
            "valid": bool(world.temporal_valid),
            "depth": int(world.temporal_depth),
            "ball_dx": float(world.temporal_ball_dx),
            "ball_dy": float(world.temporal_ball_dy),
            "active_streak": int(world.temporal_active_streak),
            "possession_streak": int(world.temporal_possession_streak),
        },
        "game_mode": {"mode": str(world.game_mode), "confidence": float(world.game_mode_confidence)},
        "possession": {
            "state": str(world.possession.state),
            "owner_track_id": int(world.possession.owner_track_id),
            "owner_team": int(world.possession.owner_team),
            "confidence": float(world.possession.confidence),
        },
        "score": {
            "for": int(world.score_for),
            "against": int(world.score_against),
            "confidence": float(world.score_confidence),
            "clock_s": float(world.clock_s),
        },
        "world_valid": bool(world.world_valid),
        "control_valid": bool(world.control_valid),
        "model_fresh": bool(world.debug.get("model_fresh", False)),
        "model_recent": bool(world.debug.get("model_recent", False)),
        "model_age_s": world.debug.get("model_age_s"),
        "model_source_frame_id": int(world.debug.get("model_source_frame_id", -1)),
        "model_inference_latency_s": float(world.debug.get("model_inference_latency_s", 0.0)),
        "score_invalid_transition": bool(world.debug.get("score_debug", {}).get("invalid_transition", False)),
        "score_invalid_transition_count": int(world.debug.get("score_invalid_transition_count", 0)),
        "model_age_at_completion_s": float(world.debug.get("model_age_at_completion_s", 0.0)),
        "model_device": str(world.debug.get("model_device", "unknown")),
        "events": {
            "goal_for": bool(world.events.goal_for),
            "goal_against": bool(world.events.goal_against),
            "ball_won": bool(world.events.ball_won),
            "ball_lost": bool(world.events.ball_lost),
        },
    }


def _rate(samples, key):
    if not samples:
        return 0.0
    return sum(1 for s in samples if s[key]) / len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Observation-only 8-10s V24 live perception stability test; NO XInput/control."
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("live_perception_check"))
    parser.add_argument("--save-samples", action="store_true", help="save PNG samples; disabled by default")
    parser.add_argument("--save-every", type=float, default=1.0)
    args = parser.parse_args()

    args.seconds = max(1.0, float(args.seconds))
    args.hz = max(1.0, float(args.hz))
    args.out.mkdir(parents=True, exist_ok=True)

    capture = WindowCapture(SETTINGS.capture)
    perception = PerceptionEngine(SETTINGS)
    samples = []
    saved = 0
    start = time.monotonic()
    next_step = start
    next_save = start

    print("[LIVE-CHECK] observation-only: capture + perception only; XInput is NOT started")
    print(f"[LIVE-CHECK] RT-DETR device={perception.model.device} model_hz={SETTINGS.player.model_hz:.1f} stale_limit={SETTINGS.player.max_model_stale_s:.2f}s")
    capture.start()
    try:
        while time.monotonic() - start < args.seconds:
            now = time.monotonic()
            if now < next_step:
                time.sleep(min(0.01, next_step - now))
                continue

            frame = capture.read()
            timestamp = time.monotonic()
            if frame is not None:
                world = perception.step(frame, timestamp=timestamp)
                sample = _world_sample(world)
                samples.append(sample)

                print(
                    "[LIVE-CHECK] "
                    f"t={timestamp-start:5.1f}s "
                    f"fresh={sample['model_fresh']} recent={sample['model_recent']} "
                    f"players={sample['players']['total']} "
                    f"ball={sample['ball']['source']}:{sample['ball']['confidence']:.2f} "
                    f"active={sample['active']['valid']}:{sample['active']['confidence']:.2f} "
                    f"designated={sample['designated']['track_id']}:{sample['designated']['confidence']:.2f} "
                    f"pos={sample['possession']['state']}:{sample['possession']['confidence']:.2f} "
                    f"score={sample['score']['for']}:{sample['score']['against']}:{sample['score']['confidence']:.2f} "
                    f"world={sample['world_valid']} control={sample['control_valid']}"
                )

                if args.save_samples and timestamp >= next_save:
                    path = args.out / f"sample_{saved:03d}.png"
                    cv2.imwrite(str(path), frame)
                    saved += 1
                    next_save += max(0.1, args.save_every)

            next_step += 1.0 / args.hz

    finally:
        perception.close()
        capture.stop()

    summary = {
        "duration_s": args.seconds,
        "requested_hz": args.hz,
        "samples": len(samples),
        "model_fresh_rate": _rate(samples, "model_fresh"),
        "model_recent_rate": _rate(samples, "model_recent"),
        "ball_valid_rate": sum(1 for s in samples if s["ball"]["valid"]) / len(samples) if samples else 0.0,
        "ball_detector_rate": sum(1 for s in samples if s["ball"]["source"] == "DETECTOR") / len(samples) if samples else 0.0,
        "ball_predicted_rate": sum(1 for s in samples if s["ball"]["predicted"]) / len(samples) if samples else 0.0,
        "active_valid_rate": sum(1 for s in samples if s["active"]["valid"]) / len(samples) if samples else 0.0,
        "players_valid_rate": sum(1 for s in samples if s["players"]["total"] >= 5) / len(samples) if samples else 0.0,
        "possession_confidence_mean": (sum(s["possession"]["confidence"] for s in samples) / len(samples)) if samples else 0.0,
        "world_valid_rate": sum(1 for s in samples if s["world_valid"]) / len(samples) if samples else 0.0,
        "control_valid_rate": sum(1 for s in samples if s["control_valid"]) / len(samples) if samples else 0.0,
        "temporal_valid_rate": sum(1 for s in samples if s["temporal"]["valid"]) / len(samples) if samples else 0.0,
        "designated_valid_rate": sum(1 for s in samples if s["designated"]["track_id"] >= 0 and s["designated"]["confidence"] >= 0.45) / len(samples) if samples else 0.0,
        "score_values": sorted({(s["score"]["for"], s["score"]["against"]) for s in samples}),
        "score_invalid_transition_count": max((s["score_invalid_transition_count"] for s in samples), default=0),
        "score_invalid_transition_samples": sum(1 for s in samples if s["score_invalid_transition"]),
        "ball_sources": {src: sum(1 for s in samples if s["ball"]["source"] == src) for src in sorted({s["ball"]["source"] for s in samples})},
        "samples_detail": samples,
    }
    (args.out / "live_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ages = [s["model_age_s"] for s in samples if s["model_age_s"] is not None]
    latencies = [s["model_inference_latency_s"] for s in samples if s["model_inference_latency_s"] > 0]
    duration = max(1e-6, args.seconds)
    runtime = perception.model.runtime_metrics()
    summary.update({
        "observed_hz": len(samples) / duration,
        "model_device": samples[-1]["model_device"] if samples else "unknown",
        "model_age_mean_s": (sum(ages) / len(ages)) if ages else None,
        "model_age_max_s": max(ages) if ages else None,
        "model_inference_latency_mean_s": (sum(latencies) / len(latencies)) if latencies else None,
        "model_inference_latency_max_s": max(latencies) if latencies else None,
        "model_completed_samples": sum(1 for s in samples if s["model_source_frame_id"] >= 0),
        "model_runtime": runtime,
        "model_error_rate": (runtime.get("errors", 0) / max(1, runtime.get("submitted", 0))),
        "model_completed": int(runtime.get("completed", 0)),
        "model_errors": int(runtime.get("errors", 0)),
        "verdict": {
            "scheduler": bool(runtime.get("errors", 0) == 0 and runtime.get("completed", 0) > 0),
            "realtime_world": bool((sum(1 for s in samples if s["world_valid"]) / max(1, len(samples))) >= 0.50),
            "control_ready": bool((sum(1 for s in samples if s["control_valid"]) / max(1, len(samples))) >= 0.50),
        },
    })

    print("[LIVE-CHECK] completed")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"samples_detail"}}, indent=2))
    print(f"[LIVE-CHECK] report={args.out / 'live_check.json'}")


if __name__ == "__main__":
    main()

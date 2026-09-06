from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    every = max(1, int(args.every))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_no = 0
    processed = 0
    start = time.monotonic()

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, _frame = cap.read()
            if not ok:
                break

            processed += 1
            if processed % max(1, args.progress_every) == 0 or processed == 1:
                elapsed = time.monotonic() - start
                rate = processed / max(elapsed, 1e-6)
                remaining_samples = max(0, (total - frame_no + every - 1) // every) if total else 0
                eta = remaining_samples / max(rate, 1e-6) if total else 0.0
                dt = every / fps
                print(
                    f"[METRICS] frame={frame_no}/{total or '?'} "
                    f"processed={processed} every={every} dt={dt:.4f}s "
                    f"rate={rate:.2f}/s ETA={eta:.1f}s"
                )

            frame_no += every
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()

    print(
        f"[METRICS] done processed={processed} "
        f"every={every} fps={fps:.3f} dt={every/fps:.6f}s"
    )


if __name__ == "__main__":
    main()

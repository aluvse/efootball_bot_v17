from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from ..config import SETTINGS
from ..perception.capture import WindowCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("live_snapshots"))
    parser.add_argument("--every", type=float, default=0.5)
    args = parser.parse_args()

    cap = WindowCapture(SETTINGS.capture)
    args.out.mkdir(parents=True, exist_ok=True)
    cap.start()
    start = time.monotonic()
    next_save = start
    index = 0
    try:
        while time.monotonic() - start < args.seconds:
            frame = cap.read()
            now = time.monotonic()
            if frame is not None and now >= next_save:
                path = args.out / f"frame_{index:04d}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"[LIVE] saved {path}")
                index += 1
                next_save += args.every
            time.sleep(0.01)
    finally:
        cap.stop()


if __name__ == "__main__":
    main()

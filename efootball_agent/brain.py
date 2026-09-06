from __future__ import annotations

import time

from .config import SETTINGS
from .runtime.engine import AgentRuntime


def main():
    """V24 is launched from brain.py only."""
    runtime = AgentRuntime(SETTINGS)
    print("=" * 64)
    print("EFOOTBALL AGENT V24.1")
    print("Architecture: perception -> world -> planner -> PPO -> safety -> XInput")
    print("Main: brain.py")
    print("Team: OUR=BLACK, OPP=WHITE")
    print("=" * 64)
    runtime.start()
    period = 1.0 / max(1.0, SETTINGS.runtime.perception_hz)
    try:
        while True:
            started = time.monotonic()
            runtime.step()
            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n[BRAIN] Ctrl+C received")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()

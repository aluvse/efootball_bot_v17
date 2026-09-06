Root-level tool launchers are provided for convenience.

Examples from the project root:
  python tools\live_perception_snapshot.py --seconds 8
  python tools\perception_probe.py datasets\efootball_real_frames\frame_00.jpg --out probe_frame00
  python tools\metrics.py your_video.mp4 --every 30 --max-frames 3000 --progress-every 100
  python tools\verify_install.py

The actual implementations remain under efootball_agent/tools/.
  python tools\live_perception_check.py --seconds 10 --hz 10 --out live_perception_check

live_perception_check.py is observation-only: it starts WindowCapture and PerceptionEngine, but never starts AgentRuntime or XInput, so it cannot send gameplay input. It preserves the perception engine state across sequential frames and writes live_check.json plus periodic sample JPEGs.

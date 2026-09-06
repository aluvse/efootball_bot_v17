# eFootball Agent V24.1 / V17 — ball/designated temporal fusion build

# eFootball Agent V24

Новая архитектура screen-based football agent, написанная заново на уровне
perception/world/control/RL contracts. Она не является копией Google Research
Football: используется GRF как источник архитектурных идей для semantic state,
action design, reward shaping и curriculum, а perception/control остаются
специфичными для eFootball.

## Запуск

```powershell
python brain.py
```

V24.1 / V16 keeps the V24 modular contracts and focuses on perception/runtime repair before PPO changes. PPO weights and hyperparameters were not tuned in this repair pass.

## Структура

- `brain.py` — единственная точка запуска.
- `efootball_agent/perception/` — window capture, UI mask, players, team, active, ball, score.
- `efootball_agent/world/` — possession, prediction, canonical world model.
- `efootball_agent/tactical/` — bootstrap planner.
- `efootball_agent/rl/` — 170D semantic encoder + structured categorical PPO.
- `efootball_agent/control/` — safety gate + XInput.
- `efootball_agent/tools/` — perception probe, live snapshots, stride metrics.
- `efootball_agent/tests/` — regression/unit tests.

## Действия

PPO выбирает только семантические действия:
`movement(9) + football_action(6) + sprint(2)`.
Кнопка Y / goalkeeper rush отсутствует в action space.

XInput mapping:
- A = pass
- B = cross
- X = shoot
- RB = call for pressure
- LB = switch player
- RT = sprint
- Y = never emitted

## PPO bootstrap

До `teacher_steps` runtime использует проверяемый heuristic planner как
bootstrap/behavior cloning teacher. После этого включается on-policy PPO.
Состояние и policy checkpoint сохраняются в `checkpoints/v24_ppo.pt`.

## Важные ограничения

- Score detection — OCR/temporal heuristic, пока не гарантирует 100% событий.
- Team identity — appearance heuristic + temporal tracker smoothing, не обученная на полном датасете eFootball модель.
- Active marker — cyan CV detector + geometry-based association.
- RT-DETR — proposal source; pipeline не делает его prediction «истиной».
- Без локальных RT-DETR weights control по умолчанию не считается достаточно надёжным.
- Полная онлайн-валидация на кадрах 300/1200/1500 невозможна из текущего архива: там нет самих кадров с этими номерами, только 24 unlabeled samples.

## Goal / score events

V24 has two event paths:
1. HUD score OCR with temporal confirmation;
2. conservative ball goal-line crossing fallback.

The fallback uses goal-mouth geometry and ball direction and is not claimed to be
100% ground truth. A score change is the preferred authoritative screen signal.

## Install (Windows / venv)

```powershell
pip install -r requirements-win.txt
```

Put the existing local RT-DETR Hugging Face model in:

```text
models/rtdetr_r50vd/
```

The V24 runtime never falls back to full-screen capture and never emits Y / goalkeeper-rush.

## Diagnostics

```powershell
python tools\verify_install.py
python tools\perception_probe.py datasets\efootball_real_frames\frame_00.jpg --out probe_frame00
python tools\live_perception_snapshot.py --seconds 8
python tools\metrics.py your_video.mp4 --every 30 --max-frames 3000 --progress-every 100

# Equivalent module form:
python -m efootball_agent.tools.verify_install
python -m efootball_agent.tools.perception_probe datasets/efootball_real_frames/frame_00.jpg --out probe_frame00
python -m efootball_agent.tools.live_perception_snapshot --seconds 8
python -m efootball_agent.tools.metrics your_video.mp4 --every 30 --max-frames 3000 --progress-every 100
pytest -q
```

The supplied archive has 24 unlabeled real frames, so it does not contain exact frame numbers 300/1200/1500. Those specific regression probes require the original video or those frames as files.

## Realtime RT-DETR acceleration — AMD/CPU contract

RT-DETR is asynchronous and never blocks the capture/perception thread. The
worker uses latest-frame semantics with a bounded pending slot, source-frame
timestamps, exact detector age, and CPU thread tuning.

This V24.1 / V16 Windows build is intentionally CPU-only for the supplied AMD host. An NVIDIA-specific runtime is not part of this contract.

Default realtime settings are tuned for the supplied Ryzen 7 5700X3D:
- RT-DETR worker: 6 Hz
- PyTorch intra-op threads: 8
- PyTorch inter-op threads: 1
- worker CPU affinity: automatic best-effort isolation to the upper logical
  CPUs
- detector input: 640x640 when supported by the local image processor
- pending model queue: 1 frame; newest frame replaces older pending work
- live test: no JPEG writes unless explicitly requested

Tesseract OCR also runs in a separate latest-frame worker in the live pipeline
so OCR subprocess latency cannot stall capture/tracking.

## V9 realtime tracker
V9 uses asynchronous RT-DETR as a global re-detector and deterministic Hungarian + constant-velocity + local Lucas-Kanade flow for short player-detection gaps. Ball tracking uses short local CV continuity plus an alpha-beta filter. This is a CPU-first realtime architecture; PPO is unchanged.


## V12 acceptance focus
V12 keeps the V11 ball/score work and fixes the next live bottleneck: persistent player tracks across async RT-DETR gaps, active-player association through short LOST intervals, and world/control validity based on bounded maintained tracks rather than detector-frame coincidence. PPO remains untrained and is not enabled for live rollout until the observation-only acceptance metrics pass.

Project entrypoint: `python brain.py` (no `main.py`).


## V14 focus
Active-player recovery now includes bounded local cyan evidence around tracked player heads, plus slower bounded confidence decay during short marker gaps. This is intended to address live active dropouts while keeping the fail-closed control gate.


## V15 architecture step

V15 adds a bounded 4-state WorldHistory, separates active/designated/possession semantics, adds coarse game_mode, introduces StickyControlState, and adds a GRF-inspired 8-channel WorldTensor encoder. PPO remains a downstream consumer and is not claimed trained.


## V17 focus

V17 keeps the V16 capture, CPU RT-DETR scheduler, player tracking, active-player
control memory, sticky control layer, WorldHistory and semantic PPO contracts.
The new step is temporal fusion of `ball -> designated player -> possession`:

- designated player uses bounded switch hysteresis, previous-player bonus, team
  continuity and movement-to-ball evidence; it is separate from the active marker
  and possession owner.
- short ball-loss gaps preserve the last possession owner with confidence decay;
  sustained competing evidence is still required to switch ownership.
- short `PREDICTED` ball states are explicitly usable by the bounded WorldState
  bridge when their confidence/age remain inside the configured limits.
- UNKNOWN ball remains fail-closed for control and never creates new attack intent.

V17 does not add a new entry point, does not add CUDA/NVIDIA dependencies, and
does not train or tune PPO.

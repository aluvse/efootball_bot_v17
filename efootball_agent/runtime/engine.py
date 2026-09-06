from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..core import Action, FootballAction, Movement, WorldState
from ..perception.capture import WindowCapture
from ..perception.pipeline import PerceptionEngine
from ..tactical.planner import TacticalPlanner
from ..rl.encoder import StateEncoder
from ..rl.ppo import PPOAgent, Rollout
from ..rl.reward import RewardEngine
from ..control.safety import SafetyGate
from ..control.xinput import XInputController


class AgentRuntime:
    """Single V24 runtime: capture -> perception -> world -> planner/PPO -> control."""

    def __init__(self, settings):
        self.settings = settings
        self.capture = WindowCapture(settings.capture)
        self.perception = PerceptionEngine(settings)
        self.planner = TacticalPlanner()
        self.encoder = StateEncoder()
        self.agent = PPOAgent(settings.ppo, self.encoder.DIM)
        self.reward = RewardEngine(settings.reward)
        self.safety = SafetyGate(settings.control)
        self.controller = XInputController(settings.control)

        self.rollout = Rollout()
        self.teacher_states = []
        self.teacher_actions = []
        self.last_world: WorldState | None = None
        self.last_action: Action | None = None
        self.last_log_prob = 0.0
        self.last_value = 0.0
        self.teacher_count = 0
        self.decisions = 0
        self.valid_decisions = 0
        self.total_reward = 0.0
        self.last_log = time.monotonic()

    def start(self):
        self.capture.start()
        self.controller.start()
        print("[RUNTIME] V24 runtime started")

    def stop(self):
        try:
            self.controller.set_action(Action())
            self.controller.stop()
        finally:
            self.capture.stop()
            self.perception.close()
        # Always persist policy on normal stop.
        try:
            self.agent.save(self.settings.ppo.checkpoint)
        except Exception as exc:
            print(f"[RUNTIME] final checkpoint failed: {exc}")
        print("[RUNTIME] stopped")

    def _teacher_phase(self):
        return (
            not self.agent.bootstrap_done
            and self.teacher_count < self.settings.ppo.teacher_steps
        )

    def _get_teacher_action(self, world):
        action = self.planner.act(world)
        # Safety gate applies equally to bootstrap control.
        return self.safety.apply(world, action)

    def _bootstrap(self, world, state):
        action = self._get_teacher_action(world)
        self.controller.set_action(action)

        if world.control_valid:
            self.teacher_states.append(state.copy())
            self.teacher_actions.append(action)
            self.teacher_count += 1
            if len(self.teacher_states) >= self.settings.ppo.teacher_batch:
                loss = self.agent.behavior_clone_step(
                    self.teacher_states,
                    self.teacher_actions,
                )
                print(
                    f"[BOOTSTRAP] steps={self.teacher_count}/"
                    f"{self.settings.ppo.teacher_steps} loss={loss:.5f}"
                )
                self.teacher_states.clear()
                self.teacher_actions.clear()

            if self.teacher_count >= self.settings.ppo.teacher_steps:
                self.agent.bootstrap_done = True
                self.agent.save(
                    self.settings.ppo.checkpoint,
                    bootstrap_done=True,
                )
                print("[BOOTSTRAP] complete -> PPO control enabled")

        return action

    def _ppo_step(self, world, state):
        # Reward for previous action from current world events.
        if (
            self.last_world is not None
            and self.last_action is not None
            and self.last_world.control_valid
        ):
            r = self.reward.compute(world)
            self.total_reward += r
            self.rollout.states.append(self.encoder.encode(self.last_world))
            self.rollout.movements.append(self.last_action.movement)
            self.rollout.football.append(self.last_action.football)
            self.rollout.sprint.append(self.last_action.sprint)
            self.rollout.rewards.append(r)
            self.rollout.dones.append(float(world.events.match_done))
            self.rollout.log_probs.append(self.last_log_prob)
            self.rollout.values.append(self.last_value)

        action, log_prob, value = self.agent.act(
            state,
            deterministic=self.settings.ppo.deterministic,
        )
        safe_action = self.safety.apply(world, action)
        self.controller.set_action(safe_action)
        self.last_action = safe_action
        self.last_log_prob = log_prob
        self.last_value = value
        self.last_world = world
        self.decisions += 1
        if world.control_valid:
            self.valid_decisions += 1

        if len(self.rollout) >= self.settings.ppo.rollout_steps:
            last_value = self.agent.net.value(
                np_to_tensor(state, self.agent.device)
            ).item()
            metrics = self.agent.update(
                self.rollout,
                float(last_value),
            )
            print(
                f"[PPO] update={metrics['update']} "
                f"reward={self.total_reward:.3f} "
                f"policy={metrics['policy']:.5f} "
                f"value={metrics['value']:.5f} "
                f"entropy={metrics['entropy']:.5f} "
                f"KL={metrics['kl']:.5f} "
                f"clip={metrics['clip']:.3f} "
                f"valid={self.valid_decisions}/{self.decisions} "
                f"score={world.score_for}:{world.score_against}"
            )
            if int(metrics["update"]) % self.settings.ppo.checkpoint_interval == 0:
                self.agent.save(self.settings.ppo.checkpoint)
            self.total_reward = 0.0
            self.decisions = 0
            self.valid_decisions = 0

        return safe_action

    def step(self):
        frame = self.capture.read()
        if frame is None:
            return None

        world = self.perception.step(
            frame,
            timestamp=time.monotonic(),
        )
        state = self.encoder.encode(world)

        if self._teacher_phase():
            action = self._bootstrap(
                world,
                state,
            )
        else:
            action = self._ppo_step(
                world,
                state,
            )

        now = time.monotonic()
        if now - self.last_log >= self.settings.runtime.log_interval_s:
            p = world.perception
            print(
                "[WORLD] "
                f"ball={world.ball.source}:{world.ball.confidence:.2f} "
                f"players={len(world.own_players)}/{len(world.opp_players)}/u{len(world.unknown_players)} "
                f"active={world.active_track_id}:{world.active_confidence:.2f} "
                f"pos={world.possession.state}:{world.possession.confidence:.2f} "
                f"mode={world.tactical.mode} "
                f"control={world.control_valid} "
                f"score={world.score_for}:{world.score_against} "
                f"events=G{int(world.events.goal_for)}-" \
                f"G{int(world.events.goal_against)} "
                f"S{int(world.events.shot_on_target)} "
                f"W{int(world.events.ball_won)} "
                f"L{int(world.events.ball_lost)} "
                f"raw={int(p.get('raw_person',0))} "
                f"validated={int(p.get('pitch_validated',0))}"
            )
            self.last_log = now

        return world


def np_to_tensor(state, device):
    import torch
    return torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

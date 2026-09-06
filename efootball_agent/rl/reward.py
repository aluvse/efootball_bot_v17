from __future__ import annotations

import numpy as np

from ..core import Team, WorldState


class RewardEngine:
    """Goal-first reward with bounded shaping."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_world: WorldState | None = None

    def reset(self):
        self.prev_world = None

    def compute(self, world: WorldState) -> float:
        if self.prev_world is None:
            self.prev_world = world
            return 0.0

        event_reward = 0.0
        event_reward += self.cfg.goal_for if world.events.goal_for else 0.0
        event_reward += self.cfg.goal_against if world.events.goal_against else 0.0
        event_reward += self.cfg.shot_on_target if world.events.shot_on_target else 0.0
        event_reward += self.cfg.ball_won if world.events.ball_won else 0.0
        event_reward += self.cfg.ball_lost if world.events.ball_lost else 0.0

        shaping = 0.0
        if world.ball_valid and self.prev_world.ball_valid:
            dx = float(world.ball.x - self.prev_world.ball.x)
            shaping += (
                dx * self.cfg.x_progress
                if dx >= 0.0
                else dx * self.cfg.x_retreat
            )
            if world.possession.owner_team == Team.OUR:
                shaping += self.cfg.possession

        shaping = float(
            np.clip(
                shaping,
                -self.cfg.shaping_clip,
                self.cfg.shaping_clip,
            )
        )

        # Event reward is allowed to dominate shaping.
        total = event_reward + shaping
        total = float(
            np.clip(
                total,
                -10.0,
                10.0,
            )
        )

        self.prev_world = world
        return total

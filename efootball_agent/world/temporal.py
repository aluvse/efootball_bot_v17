from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from ..core import Team, WorldState


@dataclass
class TemporalSample:
    timestamp: float
    ball_xy: np.ndarray
    ball_valid: bool
    active_id: int
    active_valid: bool
    designated_id: int
    designated_valid: bool
    possession_team: Team
    possession_owner_id: int
    possession_confidence: float
    world_valid: bool


class WorldHistory:
    """Short bounded temporal context for screen-based football state."""

    def __init__(self, depth: int = 4, max_ball_jump: float = 0.18):
        self.depth = max(1, int(depth))
        self.max_ball_jump = float(max_ball_jump)
        self.items: Deque[TemporalSample] = deque(maxlen=self.depth)

    def reset(self):
        self.items.clear()

    def push(self, world: WorldState):
        self.items.append(TemporalSample(
            timestamp=float(world.timestamp),
            ball_xy=np.asarray([world.ball.x, world.ball.y], np.float32),
            ball_valid=bool(world.ball.valid and not world.ball.predicted),
            active_id=int(world.active_track_id),
            active_valid=bool(world.active_valid),
            designated_id=int(world.designated_track_id),
            designated_valid=bool(world.designated_confidence >= 0.45),
            possession_team=Team(world.possession.owner_team),
            possession_owner_id=int(world.possession.owner_track_id),
            possession_confidence=float(world.possession.confidence),
            world_valid=bool(world.world_valid),
        ))

    @property
    def depth_now(self) -> int:
        return len(self.items)

    @property
    def ball_delta(self) -> tuple[float, float]:
        if len(self.items) < 2:
            return 0.0, 0.0
        a, b = self.items[-2], self.items[-1]
        if not (a.ball_valid and b.ball_valid):
            return 0.0, 0.0
        d = b.ball_xy - a.ball_xy
        return float(d[0]), float(d[1])

    @property
    def active_streak(self) -> int:
        if not self.items:
            return 0
        last = self.items[-1]
        if not last.active_valid or last.active_id < 0:
            return 0
        streak = 0
        for item in reversed(self.items):
            if item.active_valid and item.active_id == last.active_id:
                streak += 1
            else:
                break
        return streak

    @property
    def possession_streak(self) -> int:
        if not self.items:
            return 0
        last = self.items[-1]
        if last.possession_team == Team.UNKNOWN or last.possession_owner_id < 0:
            return 0
        streak = 0
        for item in reversed(self.items):
            if item.possession_team == last.possession_team and item.possession_owner_id == last.possession_owner_id:
                streak += 1
            else:
                break
        return streak

    @property
    def continuity_valid(self) -> bool:
        if len(self.items) < 2:
            return False
        a, b = self.items[-2], self.items[-1]
        if not (a.ball_valid and b.ball_valid):
            return False
        jump = float(np.linalg.norm(b.ball_xy - a.ball_xy))
        if jump > self.max_ball_jump:
            return False
        player_cont = (
            (a.active_id >= 0 and a.active_id == b.active_id)
            or (a.designated_id >= 0 and a.designated_id == b.designated_id)
            or (a.possession_owner_id >= 0 and a.possession_owner_id == b.possession_owner_id)
        )
        return bool(player_cont or (b.designated_valid and b.possession_confidence >= 0.45))

    def summary(self) -> dict[str, float | int | bool]:
        dx, dy = self.ball_delta
        return {
            "depth": self.depth_now,
            "ball_dx": dx,
            "ball_dy": dy,
            "active_streak": self.active_streak,
            "possession_streak": self.possession_streak,
            "continuity_valid": self.continuity_valid,
        }

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TrackLifecycle(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DEAD = "DEAD"


@dataclass(frozen=True)
class PlayerDetection:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float
    team: str = "UNKNOWN"
    team_confidence: float = 0.0
    role: str = "PLAYER"
    appearance: tuple[float, ...] = ()

    @property
    def footpoint(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)


@dataclass
class Track:
    track_id: int
    pos: np.ndarray
    vel: np.ndarray
    bbox: tuple[float, float, float, float] | None
    team: str = "UNKNOWN"
    team_conf: float = 0.0
    role: str = "PLAYER"
    confidence: float = 0.0
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(6, np.float32))
    age: int = 1
    hits: int = 1
    missed: int = 0
    lifecycle: TrackLifecycle = TrackLifecycle.TENTATIVE
    last_timestamp: float = 0.0
    team_history: list[tuple[str, float]] = field(default_factory=list)

    def predict(self, dt: float) -> np.ndarray:
        return self.pos + self.vel * float(dt)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class Team(IntEnum):
    UNKNOWN = 0
    OUR = 1
    OPP = 2


class Movement(IntEnum):
    IDLE = 0
    UP = 1
    UP_RIGHT = 2
    RIGHT = 3
    DOWN_RIGHT = 4
    DOWN = 5
    DOWN_LEFT = 6
    LEFT = 7
    UP_LEFT = 8


class FootballAction(IntEnum):
    NONE = 0
    PASS = 1
    CROSS = 2
    SHOOT = 3
    PRESS = 4
    SWITCH = 5


@dataclass(frozen=True)
class Action:
    movement: int = int(Movement.IDLE)
    football: int = int(FootballAction.NONE)
    sprint: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.movement), int(self.football), int(self.sprint)


@dataclass
class Player:
    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    team: Team = Team.UNKNOWN
    team_confidence: float = 0.0
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    role: str = "PLAYER"
    missed: int = 0


@dataclass
class Ball:
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    source: str = "UNKNOWN"
    predicted: bool = False
    valid: bool = False


@dataclass
class EventFlags:
    goal_for: bool = False
    goal_against: bool = False
    shot_on_target: bool = False
    ball_won: bool = False
    ball_lost: bool = False
    score_changed: bool = False
    match_done: bool = False


@dataclass
class Possession:
    state: str = "LOOSE_BALL"
    owner_team: Team = Team.UNKNOWN
    owner_track_id: int = -1
    confidence: float = 0.0
    age_s: float = 0.0


@dataclass
class TacticalState:
    mode: str = "RECOVER"
    pressure: float = 0.0
    danger: float = 0.0
    xT: float = 0.0
    future_xT: float = 0.0
    space: float = 0.0
    shot_quality: float = 0.0
    counter_score: float = 0.0


@dataclass
class WorldState:
    frame_id: int = 0
    timestamp: float = 0.0
    dt: float = 1 / 30
    ball: Ball = field(default_factory=Ball)
    own_players: list[Player] = field(default_factory=list)
    opp_players: list[Player] = field(default_factory=list)
    unknown_players: list[Player] = field(default_factory=list)
    active_track_id: int = -1
    active_player: Player | None = None
    active_confidence: float = 0.0
    control_player: Player | None = None
    designated_track_id: int = -1
    designated_player: Player | None = None
    designated_confidence: float = 0.0
    possession: Possession = field(default_factory=Possession)
    tactical: TacticalState = field(default_factory=TacticalState)
    events: EventFlags = field(default_factory=EventFlags)
    score_for: int = 0
    score_against: int = 0
    score_confidence: float = 0.0
    clock_s: float = 0.0
    game_mode: str = "UNKNOWN"
    game_mode_confidence: float = 0.0
    world_valid: bool = False
    control_valid: bool = False
    players_valid: bool = False
    ball_valid: bool = False
    active_valid: bool = False
    control_track_id: int = -1
    control_track_confidence: float = 0.0
    control_active_memory: bool = False
    possession_valid: bool = False
    temporal_valid: bool = False
    temporal_depth: int = 0
    temporal_ball_dx: float = 0.0
    temporal_ball_dy: float = 0.0
    temporal_active_streak: int = 0
    temporal_possession_streak: int = 0
    perception: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    def attack_ball(self) -> np.ndarray:
        return np.array([self.ball.x, self.ball.y], np.float32)

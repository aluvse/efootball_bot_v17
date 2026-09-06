from __future__ import annotations

from ..core import Action, FootballAction, Movement, WorldState


class SafetyGate:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, world: WorldState, action: Action) -> Action:
        if world.control_valid:
            return action
        # No random movement / shot / pass with uncertain perception.
        # Keep a neutral action until the world becomes trustworthy.
        return Action(Movement.IDLE, FootballAction.NONE, 0)

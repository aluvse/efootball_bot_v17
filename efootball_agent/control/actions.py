from __future__ import annotations

from ..core import Action, Movement


MOVEMENT_AXIS = {
    Movement.IDLE: (0.0, 0.0),
    Movement.UP: (0.0, -1.0),
    Movement.UP_RIGHT: (0.707, -0.707),
    Movement.RIGHT: (1.0, 0.0),
    Movement.DOWN_RIGHT: (0.707, 0.707),
    Movement.DOWN: (0.0, 1.0),
    Movement.DOWN_LEFT: (-0.707, 0.707),
    Movement.LEFT: (-1.0, 0.0),
    Movement.UP_LEFT: (-0.707, -0.707),
}


def to_xinput(action: Action):
    """Map high-level action to safe Xbox semantics.

    Y / goalkeeper rush is intentionally unreachable.
    """
    movement = Movement(int(action.movement))
    x, y = MOVEMENT_AXIS[movement]
    football = int(action.football)
    return {
        "lx": x,
        "ly": y,
        "pass": football == 1,
        "cross": football == 2,
        "shoot": football == 3,
        "press": football == 4,
        "switch": football == 5,
        "sprint": bool(action.sprint),
    }


class StickyControlState:
    """Temporal control adapter: policy intents become persistent stick/button state."""
    def __init__(self):
        self.current = Action()
        self.desired = Action()
        self.changed_at = 0.0

    def reset(self):
        self.current = Action()
        self.desired = Action()
        self.changed_at = 0.0

    def set_desired(self, action: Action, now: float = 0.0):
        self.desired = Action(int(action.movement), int(action.football), int(action.sprint))
        self.changed_at = float(now)

    def snapshot(self) -> Action:
        return self.desired

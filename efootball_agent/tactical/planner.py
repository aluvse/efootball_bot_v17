from __future__ import annotations

import numpy as np

from ..core import Action, FootballAction, Movement, Team, WorldState


class TacticalPlanner:
    """Deterministic football bootstrap policy.

    This is intentionally small and inspectable. It is used to bootstrap the
    policy before PPO has enough on-policy experience to be trusted.
    """

    @staticmethod
    def _movement_towards(dx: float, dy: float) -> int:
        if abs(dx) < 0.06 and abs(dy) < 0.06:
            return int(Movement.IDLE)
        sx = 1 if dx > 0.05 else -1 if dx < -0.05 else 0
        sy = 1 if dy > 0.05 else -1 if dy < -0.05 else 0
        table = {
            (0, -1): Movement.UP,
            (1, -1): Movement.UP_RIGHT,
            (1, 0): Movement.RIGHT,
            (1, 1): Movement.DOWN_RIGHT,
            (0, 1): Movement.DOWN,
            (-1, 1): Movement.DOWN_LEFT,
            (-1, 0): Movement.LEFT,
            (-1, -1): Movement.UP_LEFT,
            (0, 0): Movement.IDLE,
        }
        return int(table[(sx, sy)])

    @staticmethod
    def _nearest(players, point):
        if not players:
            return None, 1.0
        p = np.asarray(point, np.float32)
        scored = [
            (
                float(np.linalg.norm(p - np.asarray([x.x, x.y], np.float32))),
                x,
            )
            for x in players
        ]
        return min(scored, key=lambda item: (item[0], item[1].track_id))

    def act(self, world: WorldState) -> Action:
        ball = world.ball
        if not ball.valid:
            return Action(Movement.IDLE, FootballAction.NONE, 0)

        if world.possession.owner_team == Team.OUR:
            dx_goal = 1.0 - ball.x
            dy_goal = 0.5 - ball.y
            move_goal = self._movement_towards(dx_goal, dy_goal)

            if world.tactical.shot_quality >= 0.78 and ball.x >= 0.72:
                return Action(move_goal, FootballAction.SHOOT, 1)

            if world.tactical.xT >= 0.55 and world.tactical.space >= 0.35:
                if world.tactical.counter_score >= 0.62:
                    return Action(move_goal, FootballAction.PASS, 1)

            if ball.x >= 0.80 and world.tactical.space >= 0.25:
                return Action(move_goal, FootballAction.CROSS, 1)

            # Move to the best forward direction. Pass only when it can help.
            return Action(move_goal, FootballAction.PASS if world.tactical.space >= 0.42 else FootballAction.NONE, 1)

        if world.possession.owner_team == Team.OPP:
            active = world.active_player
            if active is not None:
                dx = ball.x - active.x
                dy = ball.y - active.y
            else:
                dx = 0.0
                dy = 0.0
            movement = self._movement_towards(dx, dy)
            sprint = 1 if world.tactical.pressure < 0.85 else 0
            return Action(movement, FootballAction.PRESS, sprint)

        nearest, dist = self._nearest(world.own_players, (ball.x, ball.y))
        if nearest is None:
            return Action(Movement.IDLE, FootballAction.NONE, 0)
        movement = self._movement_towards(
            ball.x - nearest.x,
            ball.y - nearest.y,
        )
        return Action(movement, FootballAction.PRESS, 1 if dist > 0.08 else 0)

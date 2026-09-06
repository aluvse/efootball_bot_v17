from __future__ import annotations

import numpy as np

from ..core import Ball, Player, Possession, TacticalState, Team
from .possession import PossessionEstimator
from .prediction import Predictor


class WorldModel:
    def __init__(self, cfg, tactical_cfg):
        self.cfg = cfg
        self.tactical_cfg = tactical_cfg
        self.possession = PossessionEstimator(cfg)
        self.predictor = Predictor(cfg.prediction_horizons)

    def reset(self):
        self.possession.reset()

    def canonical_xy(self, point):
        x, y = float(point[0]), float(point[1])
        if self.cfg.attack_direction == "left":
            x = 1.0 - x
        return np.array([x, y], dtype=np.float32)

    def canonical_player(self, player):
        player.x, player.y = tuple(self.canonical_xy((player.x, player.y)))
        if self.cfg.attack_direction == "left":
            player.vx = -float(player.vx)
        return player

    def canonical_ball(self, ball):
        x, y = self.canonical_xy((ball.x, ball.y))
        return Ball(
            x=float(x), y=float(y),
            vx=(-ball.vx if self.cfg.attack_direction == "left" else ball.vx),
            vy=float(ball.vy),
            confidence=ball.confidence, source=ball.source,
            predicted=ball.predicted, valid=ball.valid,
        )

    @staticmethod
    def xT(ball):
        x, y = float(ball.x), float(ball.y)
        central = 1.0 - abs(y - 0.5) / 0.5
        base = np.clip((x - 0.15) / 0.85, 0, 1)
        final = np.clip((x - 0.68) / 0.32, 0, 1)
        return float(
            np.clip(
                0.72 * base
                + 0.28 * final * (0.55 + 0.45 * central),
                0,
                1,
            )
        )

    @staticmethod
    def pressure(opp, ball):
        if not opp:
            return 0.0
        b = np.array([ball.x, ball.y], np.float32)
        distance = min(
            float(
                np.linalg.norm(
                    b - np.array([p.x, p.y], np.float32)
                )
            )
            for p in opp
        )
        return float(
            np.clip(
                (0.23 - distance) / 0.23,
                0,
                1,
            )
        )

    @staticmethod
    def space(own, opp):
        if not own:
            return 0.0
        values = []
        for player in own:
            if not opp:
                values.append(1.0)
                continue
            pp = np.array([player.x, player.y], np.float32)
            nearest = min(
                float(
                    np.linalg.norm(
                        pp - np.array([o.x, o.y], np.float32)
                    )
                )
                for o in opp
            )
            values.append(
                float(np.clip(nearest / 0.22, 0, 1))
            )
        return float(np.mean(values))

    def update(self, own, opp, ball, dt):
        possession = self.possession.update(
            own,
            opp,
            ball,
            dt,
        )

        pressure = self.pressure(
            opp,
            ball,
        )
        space = self.space(
            own,
            opp,
        )
        xt = self.xT(ball)

        ball_predictions = self.predictor.point(
            np.array([ball.x, ball.y], np.float32),
            np.array([ball.vx, ball.vy], np.float32),
        )

        predicted_x = max(
            (
                float(point[0])
                for point in ball_predictions.values()
            ),
            default=ball.x,
        )

        future_xt = self.xT(
            Ball(
                x=predicted_x,
                y=ball.y,
            )
        )

        shot_quality = 0.0
        if ball.x >= self.tactical_cfg.shot_x:
            shot_quality = float(
                np.clip(
                    0.45 * np.clip(
                        (ball.x - self.tactical_cfg.shot_x)
                        / max(1e-6, 1.0 - self.tactical_cfg.shot_x),
                        0,
                        1,
                    )
                    + 0.25 * (1 - abs(ball.y - 0.5) / 0.5)
                    + 0.15 * (1 - pressure)
                    + 0.15 * space,
                    0,
                    1,
                )
            )

        counter = float(
            np.clip(
                0.50 * max(ball.vx, 0.0)
                + 0.25 * (1 - pressure)
                + 0.25 * xt,
                0,
                1,
            )
        )

        danger = float(
            np.clip(
                0.75 * pressure
                + 0.25 * max(0.0, (0.60 - ball.x) / 0.60),
                0,
                1,
            )
        )

        if possession.owner_team == Team.OUR:
            mode = (
                "COUNTER_ATTACK"
                if counter >= self.tactical_cfg.counter_threshold
                else "ATTACK"
            )
        elif possession.owner_team == Team.OPP:
            mode = (
                "DEFEND"
                if danger >= self.tactical_cfg.pressure_high
                else "PRESS"
            )
        else:
            mode = "RECOVER"

        tactical = TacticalState(
            mode=mode,
            pressure=pressure,
            danger=danger,
            xT=xt,
            future_xT=float(future_xt),
            space=space,
            shot_quality=shot_quality,
            counter_score=counter,
        )

        return (
            possession,
            tactical,
            ball_predictions,
        )

    def predictions_for_players(self, players):
        result = {}
        for player in players:
            points = self.predictor.point(
                np.array([player.x, player.y], np.float32),
                np.array([player.vx, player.vy], np.float32),
            )
            for horizon, point in points.items():
                result[f"player:{player.track_id}:{horizon}"] = point
        return result

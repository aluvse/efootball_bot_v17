from __future__ import annotations

from ..core import Team, WorldState


class GameModeEstimator:
    """Conservative screen-only game-mode classifier.

    It intentionally emits only high-confidence coarse modes. A future visual
    set-piece classifier can refine this without changing the WorldState API.
    """

    def classify(self, ball_valid: bool, player_count: int, score_changed: bool, previous: str = "UNKNOWN"):
        if ball_valid and player_count >= 5:
            return "PLAY", 0.92
        if score_changed and player_count >= 5:
            return "RESTART", 0.65
        if player_count <= 2:
            return "UNKNOWN", 0.35
        if previous in {"PLAY", "RESTART"}:
            return "RESTART", 0.45
        return "UNKNOWN", 0.40

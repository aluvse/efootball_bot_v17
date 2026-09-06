from __future__ import annotations

import numpy as np

from ..core import Team, WorldState


class StateEncoder:
    """GRF-inspired semantic state with explicit temporal/designated fields."""

    DIM = 170

    @staticmethod
    def _player_features(player, limit=1.0):
        if player is None:
            return np.zeros(5, np.float32)
        return np.asarray([
            np.clip(player.x, 0.0, 1.0), np.clip(player.y, 0.0, 1.0),
            np.clip(player.vx, -limit, limit), np.clip(player.vy, -limit, limit),
            np.clip(player.confidence, 0.0, 1.0),
        ], dtype=np.float32)

    def encode(self, world: WorldState) -> np.ndarray:
        out = []
        b = world.ball
        out.extend([b.x, b.y, np.clip(b.vx, -1.0, 1.0), np.clip(b.vy, -1.0, 1.0),
                    b.confidence, float(b.valid), float(b.predicted)])

        for player, valid, track_id in ((world.control_player, world.control_track_id >= 0, world.control_track_id),
                                        (world.designated_player, world.designated_confidence >= 0.45, world.designated_track_id)):
            if player is None:
                out.extend([0,0,0,0,0,float(valid),0])
            else:
                out.extend([player.x, player.y, np.clip(player.vx,-1,1), np.clip(player.vy,-1,1),
                            player.confidence, float(valid), (max(track_id,-1)+1)/512.0])

        own = sorted(world.own_players, key=lambda p:p.track_id)[:11]
        opp = sorted(world.opp_players, key=lambda p:p.track_id)[:11]
        for players in (own, opp):
            for i in range(11):
                out.extend(self._player_features(players[i] if i < len(players) else None))

        possession_one_hot = np.zeros(5, np.float32)
        idx = {
            "OUR_CONTROL":0, "OUR_RECEIVING":1, "CONTESTED":2,
            "LOOSE_BALL":3, "OPP_CONTROL":4,
        }.get(world.possession.state, 3)
        possession_one_hot[idx] = 1.0
        out.extend(possession_one_hot.tolist())
        out.append(float(world.possession.confidence))
        out.extend([np.clip(world.score_for/10.0,0,1), np.clip(world.score_against/10.0,0,1), np.clip(world.clock_s/6000.0,0,1)])

        t=world.tactical
        out.extend([t.pressure,t.danger,t.xT,t.space,t.shot_quality,t.counter_score])
        out.extend([float(world.world_valid),float(world.control_valid),float(world.temporal_valid)])
        out.extend([float(world.temporal_depth)/4.0, np.clip(world.temporal_ball_dx,-1,1), np.clip(world.temporal_ball_dy,-1,1),
                    min(world.temporal_active_streak,4)/4.0, min(world.temporal_possession_streak,4)/4.0])
        out.append((max(world.possession.owner_track_id,-1)+1)/512.0)
        mode=np.zeros(4,np.float32)
        mode[{"PLAY":0,"RESTART":1,"PAUSE":2,"UNKNOWN":3}.get(world.game_mode,3)] = 1.0
        out.extend(mode.tolist()); out.append(float(world.game_mode_confidence))
        out.append(np.clip(len(world.unknown_players)/11.0,0,1))

        for horizon in ("0.25","0.50","1.00"):
            p=world.predictions.get(f"ball:{horizon}")
            out.extend([b.x,b.y] if p is None else [float(p[0]),float(p[1])])

        vector=np.asarray(out,dtype=np.float32)
        if vector.shape[0] < self.DIM:
            vector=np.pad(vector,(0,self.DIM-vector.shape[0]))
        elif vector.shape[0] > self.DIM:
            vector=vector[:self.DIM]
        return np.nan_to_num(vector,nan=0.0,posinf=1.0,neginf=-1.0)


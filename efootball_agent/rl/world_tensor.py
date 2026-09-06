from __future__ import annotations

import numpy as np

from ..core import Team, WorldState


class WorldTensorEncoder:
    """GRF-like spatial representation built only from canonical WorldState.

    Channels: our, opp, ball, active, designated, possession-owner, predicted-ball, danger.
    """

    CHANNELS = 8

    def __init__(self, width=48, height=32):
        self.width = int(width)
        self.height = int(height)

    def _xy(self, x, y):
        return int(np.clip(x,0,0.999999)*self.width), int(np.clip(y,0,0.999999)*self.height)

    def encode(self, world: WorldState) -> np.ndarray:
        out=np.zeros((self.CHANNELS,self.height,self.width),np.float32)
        def put(ch,x,y,value=1.0,radius=1):
            cx,cy=self._xy(x,y)
            ya=max(0,cy-radius); yb=min(self.height,cy+radius+1)
            xa=max(0,cx-radius); xb=min(self.width,cx+radius+1)
            out[ch,ya:yb,xa:xb]=np.maximum(out[ch,ya:yb,xa:xb],float(value))
        for p in world.own_players: put(0,p.x,p.y,p.confidence)
        for p in world.opp_players: put(1,p.x,p.y,p.confidence)
        if world.ball.valid: put(2,world.ball.x,world.ball.y,world.ball.confidence,1)
        if world.active_player is not None: put(3,world.active_player.x,world.active_player.y,world.active_confidence,1)
        if world.designated_player is not None: put(4,world.designated_player.x,world.designated_player.y,world.designated_confidence,1)
        owner=None
        pool=world.own_players+world.opp_players
        if world.possession.owner_track_id>=0:
            owner=next((p for p in pool if p.track_id==world.possession.owner_track_id),None)
        if owner is not None: put(5,owner.x,owner.y,world.possession.confidence,1)
        for key, p in world.predictions.items():
            if key.startswith("ball:") and p is not None and len(p) >= 2:
                put(6, float(p[0]), float(p[1]), 0.5, 1)
                break
        if world.ball.valid:
            out[7,:,:]=float(world.tactical.danger)
        return out

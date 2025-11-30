from __future__ import annotations

import numpy as np

from src.phase3 import config
from src.phase3.core.physics import step_kernel


class Engine:
    def __init__(self, mask, field, sdf, spawner, recorder):
        self.mask = mask
        self.field = field
        self.sdf = sdf
        self.spawner = spawner
        self.recorder = recorder
        n = config.AGENT_COUNT
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.active = np.zeros((n,), dtype=np.bool_)
        self.spawner.mask_shape = mask.shape
        self.spawner.respawn(self.pos, self.vel, self.active, np.arange(n))

    def step(self):
        escaped = step_kernel(
            self.pos,
            self.vel,
            self.active,
            self.field,
            self.sdf,
            config.DT,
            config.TAU,
            config.NOISE_SIGMA,
            config.V0,
            config.RESPAWN_RADIUS,
        )
        if escaped:
            self.spawner.respawn(self.pos, self.vel, self.active, np.array(escaped, dtype=np.int64))
        self.recorder.collect(self.pos, self.vel)

from typing import Tuple

import buffer

TIME_RES = 100


class TimeInformation:
    def __init__(self, buffers: buffer.BufferCollection):
        self.start_time = min([min(b.accesses.keys()) for b in buffers.values()])
        self.end_time = max([max(b.accesses.keys()) for b in buffers.values()])
        self.duration = self.end_time - self.start_time
        self.start_time = self.start_time - self.duration // 20
        self.end_time = self.end_time + self.duration // 20
        self.duration = self.end_time - self.start_time

        self.timestep_size = max(self.duration // TIME_RES, 1)
        self.timestep_count = self.duration // self.timestep_size

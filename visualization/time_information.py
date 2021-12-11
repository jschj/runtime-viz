from typing import Tuple

import buffer

TIME_RES = 100


class TimeInformation:
    def __init__(self, buffers: buffer.BufferCollection):
        self.start_time = min([min(b.accesses.keys()) for b in buffers.values()])
        self.end_time = max([max(b.accesses.keys()) for b in buffers.values()])

        self.timestep_size = max((self.end_time - self.start_time) // 100, 1)
        self.timestep_count = (self.end_time - self.start_time) // self.timestep_size

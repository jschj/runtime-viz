import zlib
from typing import Literal

TIME_RES = 50


class TimeInformation:
    def __init__(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time
        self.start_time = self.start_time - self.duration // 20
        self.end_time = self.end_time + self.duration // 20
        self.duration = self.end_time - self.start_time

        self.timestep_size = max(self.duration // TIME_RES, 1)
        self.timestep_count = self.duration // self.timestep_size

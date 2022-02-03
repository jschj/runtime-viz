import math

import numpy as np

TIME_RES = 50


class TimeInformation:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time

        self.timestep_size = max(self.duration // TIME_RES, 1)
        self.timestep_count = math.floor(self.duration / self.timestep_size + 1.0)

        print(f"Time information: start: {self.start_time} "
              f"end: {self.end_time}, "
              f"duration: {self.duration}, "
              f"timestep size: {self.timestep_size}, "
              f"timestep count: {self.timestep_count}")

    def get_timesteps(self):
        end = self.start_time + self.timestep_size * self.timestep_count
        timesteps = np.arange(self.start_time, end, self.timestep_size, )
        assert (len(timesteps) == self.timestep_count)

        return timesteps

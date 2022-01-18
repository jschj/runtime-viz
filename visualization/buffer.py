import numpy as np

import math

from time_information import TimeInformation


def _calc_heatmap_resolution(actual_res, max_res):
    """
    Calculates the heatmap resolution (hm_res).
    Guarantees that: hm_res <= max_res and actual_res % hm_res == 0.

    Problematic edge case: If actual_res is prime, hm_res == 1
    """
    if actual_res < max_res:
        return actual_res

    hm_res = max_res
    for i in range(max_res, 0, -1):
        if actual_res % i == 0:
            hm_res = i
            break

    assert hm_res <= max_res
    assert actual_res % hm_res == 0
    return hm_res


class Buffer:
    MAX_RES = 200

    def __init__(self, details: dict):
        """ Initialize buffer using detail information from JSON """

        self.name = details["name"]
        self.type_name = details["type_name"]
        self.height = details['height']
        self.dimensionality = details["type"]

        if self.dimensionality == "pitched":
            self.width = details['width']
            self.pitch = details["pitch"]
        else:
            self.width = self.pitch = 1

        # these fields are initialized in initialize_heatmap
        self.downsampling_factor = None
        self.ti = None
        self.heatmap_frames = None
        self.hm_width = None
        self.hm_height = None

        # heatmap entry with the most accesses (is updated in add_access)
        self.highest = 0

        # quick sanity checking
        assert 0 <= self.height
        assert 0 <= self.width
        assert self.name and self.name.strip()

    def initialize_heatmap(self, ti: TimeInformation):
        # calculate heatmap dimension
        self.hm_width = _calc_heatmap_resolution(self.width, Buffer.MAX_RES)
        self.hm_height = _calc_heatmap_resolution(self.height, Buffer.MAX_RES)
        print(f"Using a {self.hm_height}x{self.hm_width} heatmap for {self.height}x{self.width} buffer ({self.name}).")
        # stores how many buffer entries are represented by a single pixel in the heatmap
        self.downsampling_factor = (self.width // self.hm_width) * (self.height // self.hm_height)
        print(f'self.downsampling_factor={self.downsampling_factor}')

        # create buffer for heatmap frames
        self.ti = ti
        self.heatmap_frames = np.zeros(shape=(ti.timestep_count + 1, self.hm_width, self.hm_height))

    def add_access(self, timeframe_index: int, index: int):
        # convert 1D index to coordinates (pitch is considered in tracking code)
        x_index = index % self.width
        y_index = index // self.width

        # calculate index in frame
        x = math.floor((x_index + 0.5) / self.width * self.hm_width)
        y = math.floor((y_index + 0.5) / self.height * self.hm_height)

        # update heatmap frame
        self.heatmap_frames[timeframe_index][x][y] = \
            self.heatmap_frames[timeframe_index][x][y] + 1

        # updated self.highest
        if self.heatmap_frames[timeframe_index][x][y] > self.highest:
            self.highest = self.heatmap_frames[timeframe_index][x][y]


BufferCollection = dict[int, Buffer]
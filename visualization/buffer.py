import numpy as np

import time_information


class Buffer:
    MAX_RES = 50

    def __init__(self, details: dict):
        """ Initialize buffer using detail information from JSON """

        self.name = details["name"]
        self.type_name = details["type_name"]
        self.height = details['height']
        self.dimensionality = details["type"].lower()
        self.first_access_time = details["first_access_time"]
        self.last_access_time = details["last_access_time"]
        self.has_access = False

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

        # quick sanity checking
        assert 0 <= self.height
        assert 0 <= self.width
        assert self.width <= self.pitch
        assert self.dimensionality in ["plain", "pitched"]
        assert self.name and self.name.strip()

    def initialize_heatmap(self, ti: time_information.TimeInformation):
        # calculate heatmap dimension
        self.hm_width = self._calc_heatmap_resolution(self.width, Buffer.MAX_RES)
        self.hm_height = self._calc_heatmap_resolution(self.height, Buffer.MAX_RES)
        # stores how many buffer entries are represented by a single pixel in the heatmap
        self.downsampling_factor = (self.width // self.hm_width) * (self.height // self.hm_height)
        print(f"Using a {self.hm_height}x{self.hm_width} heatmap for {self.height}x{self.width} buffer ({self.name}) "
              f"-> downsampling factor: {self.downsampling_factor}")

        # create buffer for heatmap frames
        self.ti = ti
        self.heatmap_frames = np.zeros(shape=(ti.timestep_count, self.hm_width, self.hm_height))

    def add_access(self, timeframe_index: int, index: int):
        self.has_access = True

        # convert 1D index to coordinates (pitch is considered in tracking code)
        x_index = index % self.width
        y_index = index // self.width

        # calculate index in frame
        x = int((x_index + 0.5) / self.width * self.hm_width)
        y = int((y_index + 0.5) / self.height * self.hm_height)
        assert x < self.hm_width
        assert y < self.hm_height

        # update heatmap frame
        self.heatmap_frames[timeframe_index][x][y] += 1

    @staticmethod
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


BufferCollection = dict[int, Buffer]

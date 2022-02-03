import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import buffer


class Heatmap:

    def __init__(self, b: buffer.Buffer, ax: plt.Axes):
        """
        Precalculate and display heatmap
        :param b: Buffer to be visualized
        :param ti: time information
        :param ax: matplotlib axis to display the heatmap on
        """
        self.buffer = b

        # add a frame with only zeros in order to visualize first time step as well
        self.prefix_sums = np.append(np.zeros(shape=(1, b.hm_width, b.hm_height)), self.buffer.heatmap_frames, 0)

        # devide every heatmap entry by downsampling factor to display average access count
        # this is only relevant if downsampling is active,
        # because a single pixel will represent more than one buffer entry
        self.prefix_sums = self.prefix_sums / self.buffer.downsampling_factor

        # convert single frames to prefix sums
        print(f"Calculating prefix sums of heatmaps for buffer {b.name}...")
        for i, frame in enumerate(self.prefix_sums):
            # do not change the first frame
            if i > 0:
                self.prefix_sums[i] = self.prefix_sums[i] + self.prefix_sums[i - 1]

        img = self.calc_frame(timerange=(self.buffer.ti.start_time, self.buffer.ti.end_time))

        self.im = ax.imshow(img, vmin=0, interpolation='none')

        # cosmetics
        ax.set_title(f"{b.name} ({self.buffer.height}x{self.buffer.width}, {self.buffer.type_name})")
        ax.set_xticks(np.arange(-.5, self.buffer.hm_height + 1, 1))
        ax.set_yticks(np.arange(-.5, self.buffer.hm_width + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # this was an attempt to draw a grid into the heatmap
        # if hm_height == b.height and hm_width == b.width:
        #    ax.grid(color="w", linestyle='-', linewidth=1)

    def calc_frame(self, timerange: Tuple[int, int]):
        """
        Use precalculated prefix sums to calculate heatmap frame for given timerange fast
        :param timerange: Tuple of (start_time, end_time). Values must be relative to ti.start_time!
        """
        a, b = timerange

        # make timepoints relative to ti.start_time
        a = a - self.buffer.ti.start_time
        b = b - self.buffer.ti.start_time

        # map timepoints to corresponding timestep (might not be necessary)
        a = math.floor(a / self.buffer.ti.timestep_size)
        b = math.floor(b / self.buffer.ti.timestep_size)

        diff = self.prefix_sums[b] - self.prefix_sums[a]
        return diff

    def update(self, timerange):
        """ Callback function to update the heatmap to a given timerange. """
        self.im.set_data(self.calc_frame(timerange=timerange))

    def get_maximum(self):
        return np.ceil(np.max(self.prefix_sums))

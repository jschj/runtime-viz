import math
from typing import Tuple

import numpy as np
from tqdm import tqdm

import buffer
from time_information import TimeInformation
import matplotlib.patches as mpatches


def _calc_resolution(actual_res, max_res):
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


class Heatmap:
    MAX_RES = 200

    def __init__(self, b: buffer.Buffer, ti: TimeInformation, ax):
        """
        Precalculate and display heatmap
        :param b: Buffer to be visualized
        :param ti: time information
        :param ax: matplotlib axis to display the heatmap on
        """

        self.ti = ti

        # largest entry in heatmap
        self.highest = 0

        # heatmap dimension
        hm_width = _calc_resolution(b.width, Heatmap.MAX_RES)
        hm_height = _calc_resolution(b.height, Heatmap.MAX_RES)
        print(f"Using a {hm_height}x{hm_width} heatmap for {b.height}x{b.width} buffer ({b.name}).")
        # stores how many buffer entries are represented by a single pixel in the heatmap
        self.downsampling_factor = (b.width // hm_width) * (b.height // hm_height)
        print(f'self.downsampling_factor={self.downsampling_factor}')

        self.histogram = np.zeros(shape=(ti.timestep_count + 1,))

        # this array will contain the prefix sums of the heatmaps lateron
        # however, for now it simply contains independent frames for each timestep
        self.prefix_sums = np.zeros(shape=(ti.timestep_count + 1, hm_width, hm_height))
        # generate heatmap for each timeframe by iterating over sorted memory accesses of buffer
        print(f"Generate heatmaps for buffer {b.name}...")
        for tp in tqdm(sorted(b.accesses)):
            relative_tp = tp - ti.start_time
            frame_index = relative_tp // ti.timestep_size

            for access in b.accesses[tp]:
                # calculate index in frame
                x = math.floor((access.x_index + 0.5) / b.width * hm_width)
                y = math.floor((access.y_index + 0.5) / b.height * hm_height)

                # update heatmap frame
                self.prefix_sums[frame_index][x][y] = \
                    self.prefix_sums[frame_index][x][y] + 1

                # updated max
                if self.prefix_sums[frame_index][x][y] > self.highest:
                    self.highest = self.prefix_sums[frame_index][x][y]

                # update histogram
                self.histogram[frame_index] = self.histogram[frame_index] + 1

        # convert single frames to prefix sums
        print(f"Calculating prefix sums of heatmaps for buffer {b.name}...")
        for i, frame in enumerate(self.prefix_sums):
            # devide every heatmap entry by downsampling factor to display average access count
            # this is only relevant if downsampling is active,
            # because a single pixel will represent more than one buffer entry
            self.prefix_sums[i] = self.prefix_sums[i] / self.downsampling_factor

            # do not change the first frame
            if i > 0:
                self.prefix_sums[i] = self.prefix_sums[i] + self.prefix_sums[i - 1]

        img = self.calc_frame(timerange=(ti.start_time, ti.end_time))

        self.im = ax.imshow(img, vmin=0)

        # cosmetics
        ax.set_title(f"{b.name} ({hm_height}x{hm_width})")
        ax.set_xticks(np.arange(-.5, hm_height + 1, 1))
        ax.set_yticks(np.arange(-.5, hm_width + 1, 1))
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
        a = a - self.ti.start_time
        b = b - self.ti.start_time

        # map timepoints to corresponding timestep (might not be necessary)
        a = a // self.ti.timestep_size
        b = b // self.ti.timestep_size

        diff = self.prefix_sums[b] - self.prefix_sums[a]
        return diff

    def get_local_histogram(self) -> np.ndarray:
        """ Return histogram of memory accesses with timesteps as categories. """
        return self.histogram

    def update(self, timerange, cmap):
        """ Callback function to update the heatmap to a given timerange. """
        self.im.set_data(self.calc_frame(timerange=timerange))

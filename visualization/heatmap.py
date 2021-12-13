from typing import Tuple

import numpy as np

import buffer
from time_information import TimeInformation

MAX_RES = 100


class Heatmap:
    def __init__(self, b: buffer.Buffer, ti: TimeInformation, ax):
        """
        Precalculate and display heatmap
        :param b: Buffer to be visualized
        :param ti: time information
        :param ax: matplotlib axis to display the heatmap on
        """

        self.ti = ti

        # heatmap dimension
        hm_width = min(b.width, MAX_RES)
        hm_height = min(b.height, MAX_RES)

        self.histogram = np.zeros(shape=(ti.timestep_count + 1,))

        # this array will contain the prefix sums of the heatmaps lateron
        # however, for now it simply contains independent frames for each timestep
        self.prefix_sums = np.zeros(shape=(ti.timestep_count + 1, hm_width, hm_height))
        # generate heatmap for each timeframe by iterating over sorted memory accesses of buffer
        for tp in sorted(b.accesses):
            relative_tp = tp - ti.start_time
            frame_index = relative_tp // ti.timestep_size

            for access in b.accesses[tp]:
                # calculate index in frame
                x = round((access.x_index / b.width) * hm_width)
                y = round((access.y_index / b.height) * hm_height)

                # update heatmap frame
                self.prefix_sums[frame_index][x][y] = \
                    self.prefix_sums[frame_index][x][y] + 1

                # update histogram
                self.histogram[frame_index] = self.histogram[frame_index] + 1

        # convert single frames to prefix sums
        for i, frame in enumerate(self.prefix_sums):
            # do not change the first frame
            if i == 0:
                continue

            # accumulate every other frame with its predecessor
            self.prefix_sums[i] = self.prefix_sums[i] + self.prefix_sums[i - 1]

        img = self.calc_frame(timerange=(ti.start_time, ti.end_time))

        self.im = ax.imshow(img, vmin=0, aspect=2)

        # cosmetics
        ax.set_title(b.name)
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
        if hm_height == b.height and hm_width == b.width:
            ax.grid(color="w", linestyle='-', linewidth=1)

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

    def update(self, timerange):
        """ Callback function to update the heatmap to a given timerange. """
        self.im.set_data(self.calc_frame(timerange=timerange))

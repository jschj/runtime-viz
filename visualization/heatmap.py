from typing import Tuple

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

        # save heatmaps using prefix sum
        self.prefix_sums = []
        # iterate over sorted memory accesses of buffer
        for tp in sorted(b.accesses):
            # TODO



        # img = self.b.generate_heatmap(timerange=timerange, max_res=MAX_RES)
        ax.set_title(b.name)
        ax.set_axis_off()
        self.im = ax.imshow(img)

    def update(self, timerange):
        self.im.set_data(self.b.generate_heatmap(timerange=timerange, max_res=MAX_RES))

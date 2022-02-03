import matplotlib as mpl
import numpy as np


class ColorLegend:
    MAXIMUM_TICKS = 4

    def __init__(self, axis):
        self.axis = axis
        self.cbar = mpl.pyplot.colorbar(
            ax=axis,
            mappable=mpl.cm.ScalarMappable(norm=None, cmap=None),  # dummpy mappable
            orientation="horizontal",
            location="top",
            pad=0.5,
            label="Color legend"
        )

    def update_legend(self, cmap, clim):
        _, highest = clim
        number_of_colors = int(highest) + 1

        # apply new colormap
        self.cbar.update_normal(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, number_of_colors, clip=True), cmap=cmap))

        # draw ticks and numbers
        ticks = np.linspace(0, highest, num=number_of_colors, endpoint=True)

        # reduce the number of ticks
        while ticks.size > ColorLegend.MAXIMUM_TICKS:
            ticks = np.delete(ticks, np.arange(1, ticks.size - 1, 2))

        tick_locs = (ticks + 0.5)  # shift ticks to the middle of the color
        self.cbar.set_ticks(tick_locs)
        self.cbar.set_ticklabels([int(x) for x in ticks])

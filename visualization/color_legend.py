import matplotlib as mpl
import numpy as np


class ColorLegend:
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

        # apply new colormap
        self.cbar.update_normal(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, highest + 1, clip=False), cmap=cmap))

        # draw ticks and numbers
        number_of_ticks = max(2, min(4, highest))
        ticks = np.linspace(0, highest + 1, num=int(number_of_ticks), endpoint=False)
        if ticks[number_of_ticks - 1] != highest:
            ticks = np.append(ticks, highest)
        tick_locs = (ticks + 0.0)
        self.cbar.set_ticks(tick_locs)
        self.cbar.set_ticklabels([int(x) for x in ticks])

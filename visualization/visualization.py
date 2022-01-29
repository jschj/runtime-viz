import math

import matplotlib as mpl
import numpy as np

import buffer
import heatmap
from time_information import TimeInformation


class Visualization:
    def __init__(self, buffers: buffer.BufferCollection, histogram: np.ndarray, ti: TimeInformation):
        self.buffers = buffers
        self.histogram = histogram
        self.ti = ti

    def visualize(self):
        # calculate size of output plots
        columns = 2
        rows = math.ceil(len(self.buffers) / 2) + 1

        # Create subplots
        fig, axs = mpl.pyplot.subplots(rows, columns)
        fig.canvas.manager.set_window_title("PMPP Memory Access Visualization")
        mpl.pyplot.subplots_adjust(hspace=0.4, bottom=0.3)

        # plot heatmaps for all buffers
        heatmaps = []
        i = 0

        # largest entry in any heatmap (to determine colormap)
        for i, (_, b) in enumerate(self.buffers.items()):
            axis = mpl.pyplot.subplot(rows, columns, i + 1)
            heatmaps.append(heatmap.Heatmap(b, axis))

        highest = max([hm.get_maximum() for hm in heatmaps])

        # get discrete colormap
        cmap = mpl.pyplot.get_cmap('copper', highest)

        # apply colormap
        for hm in heatmaps:
            hm.im.set_cmap(cmap=cmap)
            hm.im.set(clim=(0, highest))

        # hide unused plots
        i = i + 1
        while i < (rows - 1) * columns:
            mpl.pyplot.subplot(rows, columns, i + 1).set_axis_off()
            i = i + 1

        # Create overview plot that ranges over all columns
        overview = mpl.pyplot.subplot(rows, 1, rows)
        end = self.ti.end_time
        if self.ti.duration % self.ti.timestep_size == 0:
            end = end + self.ti.timestep_size
        mpl.pyplot.plot(np.arange(self.ti.start_time, end, self.ti.timestep_size), self.histogram)
        mpl.pyplot.xlim(self.ti.start_time - self.ti.duration // 20, self.ti.end_time + self.ti.duration // 20)
        mpl.pyplot.title("Access histogram")
        mpl.pyplot.xlabel("Time (ns)")

        # Create the RangeSlider
        slider_ax = mpl.pyplot.axes([0.20, 0.1, 0.60, 0.03])
        slider = mpl.widgets.RangeSlider(slider_ax, "Time selection",
                                         valmin=self.ti.start_time,
                                         valmax=self.ti.end_time,
                                         valstep=self.ti.timestep_size,
                                         valinit=(self.ti.start_time, self.ti.end_time))
        slider.valtext.set_visible(False)

        # Create limit lintes
        lower_limit_line = overview.axvline(slider.val[0], color='k')
        upper_limit_line = overview.axvline(slider.val[1], color='k')

        # create color legend
        cbar = mpl.pyplot.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, highest), cmap=cmap),
                                   ax=overview,
                                   orientation="horizontal",
                                   location="top",
                                   pad=0.5,
                                   label="Color legend"
                                   )

        number_of_ticks = min(4, highest)
        ticks = np.linspace(0, highest, num=int(number_of_ticks), endpoint=True)
        tick_locs = (ticks + 0.0)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([int(x) for x in ticks])

        def update(val):
            """ Callback function when the slider is moved"""
            for h in heatmaps:
                h.update(val)

            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()

        # register callback
        slider.on_changed(update)

        # show plot window
        mpl.pyplot.show()

        print("Goodbye!")

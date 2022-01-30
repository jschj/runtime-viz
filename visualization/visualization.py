import math
from typing import Dict

import matplotlib as mpl
import numpy as np

import buffer
import heatmap
from color_legend import ColorLegend
from time_information import TimeInformation


class Visualization:
    def __init__(self, buffers: buffer.BufferCollection, histogram: np.ndarray, ti: TimeInformation):
        self.buffers = buffers
        self.histogram = histogram
        self.ti = ti

        self.heatmaps: Dict[mpl.axis.Axis, heatmap.Heatmap] = {}

        # these fields will be initialized in visualize()
        self.fig: mpl.Figure = None
        self.overview_lower_line = None
        self.overview_upper_line = None
        self.cmap = None
        self.clim = None
        self.color_legend = None

    # callback function
    # =================
    def slider_callback(self, new_range):
        """ Callback function when the slider is moved"""
        for _, h in self.heatmaps.items():
            h.update(new_range)

        self.overview_lower_line.set_xdata([new_range[0], new_range[0]])
        self.overview_upper_line.set_xdata([new_range[1], new_range[1]])

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def key_callback(self, keypress_event: mpl.backend_bases.KeyEvent):
        if keypress_event.key == "r":
            self.reset_colormap()

    def mouse_callback(self, mouse_event: mpl.backend_bases.MouseEvent):
        axis = mouse_event.inaxes
        if axis is not None and axis in self.heatmaps:
            highest = self.heatmaps[axis].get_maximum()
            self.apply_colormap(highest=highest)

    def apply_colormap(self, highest: int):
        # get discrete colormap
        self.cmap = mpl.pyplot.get_cmap('copper', highest + 1)
        self.clim = (0, highest)

        # update heatmaps
        for _, hm in self.heatmaps.items():
            hm.im.set_cmap(cmap=self.cmap)
            hm.im.set(clim=self.clim)

        # update color legend
        self.color_legend.update_legend(cmap=self.cmap, clim=self.clim)

        self.fig.canvas.draw_idle()

    def reset_colormap(self):
        # determine maximum of highest heatmap entry over all heatmaps
        highest = max([hm.get_maximum() for _, hm in self.heatmaps.items()])

        self.apply_colormap(highest=highest)

    def visualize(self):
        # calculate size of output plots
        columns = 2
        rows = math.ceil(len(self.buffers) / 2) + 1

        # Create subplots
        self.fig, axs = mpl.pyplot.subplots(rows, columns)
        self.fig.canvas.manager.set_window_title("PMPP Memory Access Visualization")
        mpl.pyplot.subplots_adjust(hspace=0.4, bottom=0.3)

        # plot heatmaps for all buffers
        # largest entry in any heatmap (to determine colormap)
        i = 0
        for i, (_, b) in enumerate(self.buffers.items()):
            axis = mpl.pyplot.subplot(rows, columns, i + 1)
            self.heatmaps[axis] = heatmap.Heatmap(b, axis)

        # hide unused plots
        i = i + 1
        while i < (rows - 1) * columns:
            mpl.pyplot.subplot(rows, columns, i + 1).set_axis_off()
            i = i + 1

        # Create overview plot that ranges over all columns
        overview = mpl.pyplot.subplot(rows, 1, rows)
        mpl.pyplot.plot(np.arange(self.ti.start_time, self.ti.end_time, self.ti.timestep_size), self.histogram)
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
        self.overview_lower_line = overview.axvline(slider.val[0], color='k')
        self.overview_upper_line = overview.axvline(slider.val[1], color='k')

        # create color legend
        self.color_legend = ColorLegend(axis=overview)

        # initialize colormap
        self.reset_colormap()

        # register callbacks
        slider.on_changed(lambda val: self.slider_callback(val))
        self.fig.canvas.mpl_connect('key_press_event', lambda event: self.key_callback(keypress_event=event))
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.mouse_callback(mouse_event=event))

        # show plot window
        mpl.pyplot.show()

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider

import input
from heatmap import Heatmap
import time_information

if __name__ == '__main__':
    # read input
    buffers = input.read_input("testinput/test02.json")
    ti = time_information.TimeInformation(buffers)

    # calculate size of output plots
    columns = 2
    rows = math.ceil(len(buffers) / 2) + 1

    # Create subplots
    fig, axs = plt.subplots(rows, columns)
    fig.canvas.manager.set_window_title("PMPP Memory Access Visualization")
    plt.subplots_adjust(hspace=0.4, bottom=0.2)

    # plot heatmaps for all buffers
    heatmaps = []
    i = 0
    for i, (_, b) in enumerate(buffers.items()):
        axis = plt.subplot(rows, columns, i + 1)
        hm = Heatmap(b, ti, axis)
        heatmaps.append(hm)

    # hide unused plots
    while i < (rows - 1) * columns:
        plt.subplot(rows, columns, i + 1).set_axis_off()
        i = i + 1

    # Create overview plot that ranges over all columns
    overview = plt.subplot(rows, 1, rows)
    plt.plot(np.arange(ti.start_time, ti.end_time, 0.01), np.cos(np.arange(ti.start_time, ti.end_time, 0.01)))  # TODO: dummy plot

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Range",
                         valmin=ti.start_time,
                         valmax=ti.end_time,
                         valstep=ti.timestep_size,
                         valinit=(ti.start_time, ti.end_time))

    # Create limit lintes
    lower_limit_line = overview.axvline(slider.val[0], color='k')
    upper_limit_line = overview.axvline(slider.val[1], color='k')


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
    plt.show()

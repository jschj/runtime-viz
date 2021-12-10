import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider

import input
import preprocessing
from heatmap import Heatmap

if __name__ == '__main__':
    # read input
    buffers = input.read_input("testinput/test01.json")
    pre = preprocessing.Preprocessing(buffers)

    # calculate basic time values
    start_time, end_time = pre.get_time_range()
    if end_time <= start_time:
        end_time = start_time + 1
    timestep_size = max((end_time - start_time) // 100, 1)
    timestep_count = (end_time - start_time) // timestep_size

    # calculate size of output plots
    columns = 2
    rows = math.ceil(pre.number_of_buffers() / 2) + 1

    # Create subplots
    fig, axs = plt.subplots(rows, columns)
    fig.canvas.manager.set_window_title("PMPP Memory Access Visualization")
    plt.subplots_adjust(hspace=0.4, bottom=0.2)

    # plot heatmaps for all buffers
    heatmaps = []
    i = 0
    for i, (_, b) in enumerate(buffers.items()):
        axis = plt.subplot(rows, columns, i + 1)
        hm = Heatmap(b, (start_time, end_time), axis)
        heatmaps.append(hm)

    # hide unused plots
    while i < (rows - 1) * columns:
        plt.subplot(rows, columns, i + 1).set_axis_off()
        i = i + 1

    # Create overview plot that ranges over all columns
    overview = plt.subplot(rows, 1, rows)
    plt.plot(np.arange(start_time, end_time, 0.01), np.cos(np.arange(start_time, end_time, 0.01)))  # TODO: dummy plot

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Range",
                         valmin=start_time,
                         valmax=end_time,
                         valstep=timestep_size,
                         valinit=(start_time, end_time))

    # Create limit lintes
    lower_limit_line = overview.axvline(slider.val[0], color='k')
    upper_limit_line = overview.axvline(slider.val[1], color='k')


    def update(val):
        """ Callback function when the slider is moved"""
        for h in heatmaps:
            h.update(val)

        hm.update(timerange=val)

        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()


    # register callback
    slider.on_changed(update)

    # show plot window
    plt.show()

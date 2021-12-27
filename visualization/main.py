import math
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import RangeSlider

import input
from heatmap import Heatmap
import time_information

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(__file__)} <path to input file (.json or .bson)>")
        exit(255)

    inputfilepath = sys.argv[1]
    # read input
    buffers = input.read_input(inputfilepath)
    ti = time_information.TimeInformation(buffers)

    # calculate size of output plots
    columns = 2
    rows = math.ceil(len(buffers) / 2) + 1

    # Create subplots
    fig, axs = plt.subplots(rows, columns)
    fig.canvas.manager.set_window_title("PMPP Memory Access Visualization")
    plt.subplots_adjust(hspace=0.4, bottom=0.3)

    # plot heatmaps for all buffers
    heatmaps = []
    i = 0
    global_histogram = np.zeros(shape=(ti.timestep_count + 1,))
    for i, (_, b) in enumerate(buffers.items()):
        axis = plt.subplot(rows, columns, i + 1)
        hm = Heatmap(b, ti, axis)
        global_histogram = global_histogram + hm.get_local_histogram()
        heatmaps.append(hm)

    # hide unused plots
    i = i + 1
    while i < (rows - 1) * columns:
        plt.subplot(rows, columns, i + 1).set_axis_off()
        i = i + 1

    # Create overview plot that ranges over all columns
    overview = plt.subplot(rows, 1, rows)
    end = ti.end_time
    if ti.duration % ti.timestep_size == 0:
        end = end + ti.timestep_size
    plt.plot(np.arange(ti.start_time, end, ti.timestep_size), global_histogram)
    plt.xlim(ti.start_time - ti.duration // 20, ti.end_time + ti.duration // 20)

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Range",
                         valmin=ti.start_time,
                         valmax=ti.end_time,
                         valstep=ti.timestep_size,
                         valinit=(ti.start_time, ti.end_time))
    slider.valtext.set_visible(False)

    # Create limit lintes
    lower_limit_line = overview.axvline(slider.val[0], color='k')
    upper_limit_line = overview.axvline(slider.val[1], color='k')


    def update(val):
        """ Callback function when the slider is moved"""
        # print(f"Slider moved! New timerange: {val[0]} - {val[1]}")

        for h in heatmaps:
            h.update(val)

        # print("Finished calculating new heatmaps.")

        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()


    # register callback
    slider.on_changed(update)

    # show plot window
    plt.show()

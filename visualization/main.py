import math
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

    # largest entry in any heatmap (to determine colormap)
    max_number = 0
    for i, (_, b) in enumerate(buffers.items()):
        axis = plt.subplot(rows, columns, i + 1)
        hm = Heatmap(b, ti, axis)

        # update hightest
        if hm.highest > max_number:
            max_number = hm.highest

        global_histogram = global_histogram + hm.get_local_histogram()
        heatmaps.append(hm)

    max_number = max_number + 1

    # get discrete colormap
    cmap = plt.get_cmap('copper', max_number)

    # apply colormap
    for hm in heatmaps:
        hm.im.set_cmap(cmap=cmap)
        hm.im.set(clim=(0, max_number))

    print("Show visualization window...")

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
    plt.title("Access histogram")
    plt.xlabel("Time (ns)")

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Time selection",
                         valmin=ti.start_time,
                         valmax=ti.end_time,
                         valstep=ti.timestep_size,
                         valinit=(ti.start_time, ti.end_time))
    slider.valtext.set_visible(False)

    # Create limit lintes
    lower_limit_line = overview.axvline(slider.val[0], color='k')
    upper_limit_line = overview.axvline(slider.val[1], color='k')

    # create color legend
    cbar = plt.colorbar(mappable=ScalarMappable(norm=Normalize(0, max_number), cmap=cmap),
                        ax=overview,
                        orientation="horizontal",
                        location="top",
                        pad=0.5,
                        label="Color legend"
                        )

    number_of_ticks = min(4, max_number)
    ticks = np.linspace(0, max_number, num=int(number_of_ticks), endpoint=True)
    tick_locs = (ticks + 0.5)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([int(x) for x in ticks])

    def update(val):
        """ Callback function when the slider is moved"""
        # print(f"Slider moved! New timerange: {val[0]} - {val[1]}")

        for h in heatmaps:
            h.update(val, cmap=cmap)

        # print("Finished calculating new heatmaps.")

        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()


    # register callback
    slider.on_changed(update)

    # show plot window
    plt.show()

    print("Goodbye!")

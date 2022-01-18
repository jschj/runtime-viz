import math
import sys
import os
import time

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
    start_wclock = time.time() # for performance measurement
    if len(sys.argv) != 5:
        print(f"Usage: {os.path.basename(__file__)} <buffer file> <access file> <start time> <end time>")
        exit(255)

    buffer_filepath = sys.argv[1]
    access_filepath = sys.argv[2]
    start_time = int(sys.argv[3])
    end_time = int(sys.argv[4])

    # read input
    buffers = input.init_buffers(buffer_filepath)
    ti = time_information.TimeInformation(start_time, end_time)
    for _, b in buffers.items():
        b.initialize_heatmap(ti)
    histogram = input.process_accesses(buffers, access_filepath, ti)

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

    # largest entry in any heatmap (to determine colormap)
    for i, (_, b) in enumerate(buffers.items()):
        axis = plt.subplot(rows, columns, i + 1)
        heatmaps.append(Heatmap(b, axis))

    highest = max([hm.get_maximum() for hm in heatmaps])

    # get discrete colormap
    cmap = plt.get_cmap('copper', highest)

    # apply colormap
    for hm in heatmaps:
        hm.im.set_cmap(cmap=cmap)
        hm.im.set(clim=(0, highest))

    print(f"Setup took {time.time() - start_wclock:.2f} seconds. Showing visualization window...")

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
    plt.plot(np.arange(ti.start_time, end, ti.timestep_size), histogram)
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
    cbar = plt.colorbar(mappable=ScalarMappable(norm=Normalize(0, highest), cmap=cmap),
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
    plt.show()

    print("Goodbye!")

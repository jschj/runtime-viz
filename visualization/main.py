import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

import input
import preprocessing
from heatmap import Heatmap

if __name__ == '__main__':
    buffers = input.read_input("testinput/test02.json")
    pre = preprocessing.Preprocessing(buffers)

    start_time, end_time = pre.get_time_range()
    if end_time <= start_time:
        end_time = start_time + 1
    timestep_size = max((end_time - start_time) // 100, 1)
    timestep_count = (end_time - start_time) // timestep_size

    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(bottom=0.25)

    hm = Heatmap(buffers[0], (start_time, end_time), axs[0][0])

    # Create the RangeSlider
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Range",
                         valmin=start_time,
                         valmax=end_time,
                         valstep=timestep_size,
                         valinit=(start_time, end_time))

    def update(val):
        hm.update(timerange=val)
        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

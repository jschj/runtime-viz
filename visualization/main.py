import os
import sys
import time

import input
import time_information
import visualization

if __name__ == '__main__':
    start_wclock = time.time()  # for performance measurement
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(__file__)} <buffer file>")
        exit(255)

    buffer_filepath = sys.argv[1]

    # read input
    buffers, access_filepaths = input.init_buffers_file(buffer_filepath)
    earliest_access_time = min([b.first_access_time for _, b in buffers.items()])
    latest_access_time = max([b.last_access_time for _, b in buffers.items()])
    ti = time_information.TimeInformation(earliest_access_time, latest_access_time)
    for _, b in buffers.items():
        b.initialize_heatmap(ti)

    histogram = input.process_accesses(buffers, os.path.join(os.path.dirname(buffer_filepath), access_filepaths[0]), ti)

    print(f"Setup took {time.time() - start_wclock:.2f} seconds. Showing visualization window...")

    vis = visualization.Visualization(buffers, histogram, ti)
    vis.visualize()

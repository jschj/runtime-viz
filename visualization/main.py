import os
import sys
import time
import argparse

import input
import time_information
import visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive visualiazation of CUDA memory accesses in GPU global '
                                                 'memory. ')

    parser.add_argument("buffers_filepath", metavar="\"filepath to buffers.json\"", type=str)

    args = parser.parse_args()
    print(args.buffers_filepath)

    start_wclock = time.time()  # for performance measurement

    # read input
    buffers, access_filepaths = input.init_buffers_file(args.buffers_filepath)
    earliest_access_time = min([b.first_access_time for _, b in buffers.items()])
    latest_access_time = max([b.last_access_time for _, b in buffers.items()])
    ti = time_information.TimeInformation(earliest_access_time, latest_access_time)
    for _, b in buffers.items():
        b.initialize_heatmap(ti)

    histogram = input.process_accesses(buffers,
                                       os.path.join(os.path.dirname(args.buffers_filepath), access_filepaths[0]), ti)

    print(f"Setup took {time.time() - start_wclock:.2f} seconds. Showing visualization window...")

    vis = visualization.Visualization(buffers, histogram, ti)
    vis.visualize()

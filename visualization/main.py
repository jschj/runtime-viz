import argparse
import os
import re
import time

import numpy as np

import input
import time_information
import visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive visualiazation of CUDA memory accesses in GPU global '
                                                 'memory. ')

    parser.add_argument("filepath",
                        metavar="filepath",
                        help="Filepath to buffers.json",
                        type=str)
    parser.add_argument("-a", "--access-file-regex",
                        default=".*", type=str,
                        required=False,
                        dest="access_file_regex",
                        metavar="regex",
                        help="Only access files with a name matching this regular expression will be considered. "
                             "Example: -a \".*gmem_kernel.*\"")
    parser.add_argument("-p", "--buffer-positive-regex",
                        default=".*",
                        type=str,
                        required=False,
                        dest="buffer_positive_regex",
                        metavar="regex",
                        help="Only buffers with a name matching this regular expression will be visualized. "
                             "Example: -p \"vector01|vector02\"")
    parser.add_argument("-n", "--buffer-negative-regex",
                        default="(?!x)x",  # guaranteed never to match anything
                        type=str,
                        required=False,
                        dest="buffer_negative_regex",
                        metavar="regex",
                        help="Buffers with a name matching this regular expression will *not* be visualized. "
                             "Example: -n \"vector03|vector04\"")

    args = parser.parse_args()

    # for performance measurement
    start_wclock = time.time()

    # read buffers.json and init time infromation and heatmaps
    buffers, access_filepaths = input.init_buffers_file(args.filepath)
    earliest_access_time = min([b.first_access_time for _, b in buffers.items()])
    latest_access_time = max([b.last_access_time for _, b in buffers.items()])
    ti = time_information.TimeInformation(earliest_access_time, latest_access_time)
    for _, b in buffers.items():
        b.initialize_heatmap(ti)

    # initialize histogram
    histogram = np.zeros(shape=(ti.timestep_count,))

    # read access files ("xy.accesses.bin") which match regex
    for filename in access_filepaths:
        if re.fullmatch(args.access_file_regex, filename) is not None:
            input.process_accesses(buffers=buffers,
                                   access_filepath=os.path.join(os.path.dirname(args.filepath), filename),
                                   ti=ti,
                                   histogram=histogram)
        else:
            print(
                f"Ignoring access information file {filename}, because its name does not match the access file regex.")

    # filter buffers
    for key, b in list(buffers.items()):
        if re.fullmatch(args.buffer_positive_regex, b.name) is None:
            print(f"Buffer {b.name} is not visualized, because its named does not match the buffer positive regex.")
            del buffers[key]

        elif re.fullmatch(args.buffer_negative_regex, b.name) is not None:
            print(f"Buffer {b.name} is not visualized, because its named does match the buffer negative regex.")
            del buffers[key]

        elif not b.has_access:
            print(f"Buffer {b.name} is not visualized, because it contains no accesses.")
            del buffers[key]

    if len(buffers) == 0:
        print("There are no buffers to visualize :/")
        exit(1)

    # print setup timing
    print(f"Setup took {time.time() - start_wclock:.2f} seconds. Showing visualization window...")

    vis = visualization.Visualization(buffers, histogram, ti)
    vis.visualize()

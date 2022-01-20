import json
import zlib
import struct

import numpy as np

import buffer
import time_information


def init_buffers(buffer_filepath: str) -> buffer.BufferCollection:
    buffers = {}
    print("Parsing input file and initializing buffers...")

    # read and parse input file
    with open(buffer_filepath, 'r') as inputfile:
        text = inputfile.read()
        content = json.loads(text)

    # initialize buffers
    for bufferdetails in content["buffers"]:
        identifier = bufferdetails["id"]
        b = buffer.Buffer(details=bufferdetails)
        buffers[identifier] = b

    return buffers


def process_accesses(buffers: buffer.BufferCollection, access_filepath: str, ti: time_information.TimeInformation):
    """
    :param buffers: BufferCollection with buffers. This function will register accesses to the corresponding buffer.
    :param access_filepath: Filepath to zlib-compressed binary file containing access information.
    :param ti: Time Info
    :return:
    """
    chunk_size: int = 16*1024  # 16 KiB
    line_width: int = 13  # the number of bytes representing a single access

    # initialize histogram
    histogram = np.zeros(shape=(ti.timestep_count + 1,))

    print("Reading and processing access information file...")

    with open(access_filepath, 'rb') as access_file:
        dco = zlib.decompressobj(wbits=zlib.MAX_WBITS | 32)  # automatic header detection
        buf = b''  # start with empty buffer
        while not dco.eof:
            # refill buffer from file if necessary
            if len(buf) < chunk_size:
                buf = buf + access_file.read(chunk_size)

            decompressed_data = dco.decompress(buf, max_length=line_width*(chunk_size // line_width))
            # unpack data from binary
            for bufferid, timestamp, index in struct.iter_unpack("<BQL", decompressed_data):
                # calculate frame index (time domain)
                relative_tp = timestamp - ti.start_time
                frame_index = relative_tp // ti.timestep_size

                # update histogram
                histogram[frame_index] += 1

                # register access in correct buffer
                # TODO: error handling
                buffers[bufferid].add_access(timeframe_index=frame_index, index=index)

            buf = dco.unconsumed_tail

    return histogram

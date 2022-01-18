import json
import zlib
from typing import Literal

import numpy as np

import buffer
from time_information import TimeInformation


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


def process_accesses(buffers: buffer.BufferCollection, access_filepath: str, ti: TimeInformation):
    """
    :param buffers: BufferCollection with buffers. This function will register accesses to the corresponding buffer.
    :param access_filepath: Filepath to zlib-compressed binary file containing access information.
    :param ti: Time Info
    :return:
    """
    chunk_size: int = 1024  # 1 KiB
    line_width: int = 13  # the number of bytes representing a single access
    endianness: Literal['little', 'big'] = 'little'

    # initialize histogram
    histogram = np.zeros(shape=(ti.timestep_count + 1,))

    print("Reading and processing access information file...")

    def process(data):
        assert len(data) == line_width

        # deserialize information from bytes
        bufferid = int.from_bytes(data[0:1], byteorder=endianness, signed=False)
        timestamp = int.from_bytes(data[1:9], byteorder=endianness, signed=False)
        index = int.from_bytes(data[9:13], byteorder=endianness, signed=False)

        # calculate frame index (time domain)
        relative_tp = timestamp - ti.start_time
        frame_index = relative_tp // ti.timestep_size

        # update histogram
        histogram[frame_index] = histogram[frame_index] + 1

        # register access in correct buffer
        # TODO: error handling
        buffers[bufferid].add_access(timeframe_index=frame_index, index=index)

    with open(access_filepath, 'rb') as access_file:
        dco = zlib.decompressobj(wbits=zlib.MAX_WBITS | 32)  # automatic header detection
        buf = b''  # start with empty buffer
        while not dco.eof:
            # refill buffer from file if necessary
            if len(buf) < chunk_size:
                buf = buf + access_file.read(chunk_size)

            decompressed_data = dco.decompress(buf, max_length=line_width)
            process(decompressed_data)
            buf = dco.unconsumed_tail

    return histogram

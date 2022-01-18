import os
import sys
import zlib
from typing import Literal

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(__file__)} <access file>")
        exit(255)

    access_filepath = sys.argv[1]

    chunk_size: int = 1024  # 1 KiB
    line_width: int = 13  # the number of bytes representing a single access
    endianness: Literal['little', 'big'] = 'little'

    start_time = sys.maxsize
    end_time = 0

    with open(access_filepath, 'rb') as access_file:
        dco = zlib.decompressobj(wbits=zlib.MAX_WBITS | 32)  # automatic header detection
        buf = b''  # start with empty buffer
        while not dco.eof:
            # refill buffer from file if necessary
            if len(buf) < chunk_size:
                buf = buf + access_file.read(chunk_size)

            decompressed_data = dco.decompress(buf, max_length=line_width)

            # process data
            timestamp = int.from_bytes(decompressed_data[1:9], byteorder=endianness, signed=False)
            if timestamp < start_time:
                start_time = timestamp
            if timestamp > end_time:
                end_time = timestamp
            # ========

            buf = dco.unconsumed_tail

    print(f"start time:\t {start_time}")
    print(f"end time:\t {end_time}")

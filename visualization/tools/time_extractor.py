import os
import sys
import zlib
import time
from typing import Literal

if __name__ == '__main__':
    start_wtime = time.time()

    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(__file__)} <access file>")
        exit(255)

    access_filepath = sys.argv[1]

    chunk_size: int = 16*1024  # 16 KiB
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

            decompressed_data = dco.decompress(buf, max_length=line_width*(chunk_size // line_width))
            assert len(decompressed_data) % line_width == 0
            for i in range(len(decompressed_data) // line_width):
                offset = i * line_width
                # process data
                timestamp = int.from_bytes(decompressed_data[offset + 1: offset + 9], byteorder=endianness, signed=False)
                if timestamp < start_time:
                    start_time = timestamp
                if timestamp > end_time:
                    end_time = timestamp
            # ========
            buf = dco.unconsumed_tail

    print(f"start time:\t {start_time}")
    print(f"end time:\t {end_time}")
    print(f"execution took {time.time() - start_wtime:.2f} seconds")

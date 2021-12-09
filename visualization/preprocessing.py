import buffer


class Preprocessing:
    def __init__(self, buffers: buffer.BufferCollection):
        self.buffers = buffers

    def number_of_buffers(self):
        return len(self.buffers)

    def get_time_range(self):
        earliest = 10_000_000
        latest = 0

        for _, b in self.buffers.items():
            for access in b.accesses:
                time = access.time
                if time < earliest:
                    earliest = time
                if time > latest:
                    latest = time

        return earliest, latest

import numpy as np

# Divisor which is applied to all timepoints
TIME_DIVISOR = 1


class MemoryAccess:
    def __init__(self, x_index, y_index):
        self.x_index = x_index
        self.y_index = y_index


class Buffer:
    """ Abstract representation of a buffer """

    def __init__(self, details: dict):
        """ Initialize buffer using detail information from BSON """
        self.name = details["name"]

        # dict containing memory accesses (key: time, entry: list of accesses at that time
        self.accesses = {}
        self.height = None
        self.width = None

    def add_memory_access(self, details: dict):
        """ Register a new memory access using detail information from BSON """
        pass

    def sanity_checks(self):
        """ Run basic consistency checks on buffer and memory access data """
        assert 0 <= self.height
        assert 0 <= self.width
        assert self.name and self.name.strip()

        for _, access_list in self.accesses.items():
            for a in access_list:
                assert 0 <= a.x_index < self.width
                assert 0 <= a.y_index < self.height


BufferCollection = dict[int, Buffer]


class Buffer1D(Buffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details['height']
        self.width = 1

    def add_memory_access(self, details: dict):
        time = details['t'] // TIME_DIVISOR
        index = details['i']
        x_index = 0
        y_index = index
        ma = MemoryAccess(x_index, y_index)

        if time not in self.accesses:
            self.accesses[time] = []
        self.accesses[time].append(ma)


class Buffer2D(Buffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]
        self.width = details["width"]
        self.pitch = details["pitch"]

    def add_memory_access(self, details: dict):
        time = details['t'] // TIME_DIVISOR
        index = details['i']

        # pitch is considered in tracking code
        x_index = index % self.width
        y_index = index // self.width
        ma = MemoryAccess(x_index, y_index)

        if time not in self.accesses:
            self.accesses[time] = []
        self.accesses[time].append(ma)

    def sanity_checks(self):
        super().sanity_checks()
        assert self.width <= self.pitch

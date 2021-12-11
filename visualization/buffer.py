import numpy as np


class MemoryAccess:
    def __init__(self, x_index, y_index):
        self.x_index = x_index
        self.y_index = y_index;


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
        assert 0 <= self.width <= self.pitch
        assert self.name and self.name.strip()

        for _, access_list in self.accesses.items():
            for a in access_list:
                assert 0 <= a.x_index < self.width
                assert 0 <= a.y_index < self.height

    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        """ Calculate heatmap for given timerange.
        If one or more dimension is larger than max_res, memory accesses are mapped onto the interval [0, max_res)
        """
        pass


BufferCollection = dict[int, Buffer]


class Buffer1D(Buffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details['height']
        self.width = 1

    def add_memory_access(self, details: dict):
        time = details['t']
        index = details['i']
        x_index = 0
        y_index = index
        ma = MemoryAccess(x_index, y_index)

        if time not in self.accesses:
            self.accesses[time] = []
        self.accesses[time].append(ma)

    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        start_time, end_time = timerange

        w = 1
        h = min(self.height, max_res)
        shape = (h, w)

        img = np.zeros(shape=shape)

        for access in self.accesses:
            if start_time <= access.time <= end_time:
                x = 0
                y = int((access.y_index / self.height) * h)
                img[y][x] = img[y][x] + 1

        return img


class Buffer2D(Buffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]
        self.width = details["width"]
        self.pitch = details["pitch"]

    def add_memory_access(self, details: dict):
        time = details['t']
        index = details['i']
        x_index = index // self.pitch
        y_index = index % self.pitch
        ma = MemoryAccess(x_index, y_index)

        if time not in self.accesses:
            self.accesses[time] = []
        self.accesses[time].append(ma)

    def sanity_checks(self):
        super().sanity_checks()
        assert self.width <= self.pitch

    # @functools.lru_cache(maxsize=100)
    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        start_time, end_time = timerange

        w = min(self.width, max_res)
        h = min(self.height, max_res)
        shape = (h, w)

        img = np.zeros(shape=shape)

        for access in self.accesses:
            if start_time <= access.time <= end_time:
                x = int((access.x_index / self.width) * w)
                y = int((access.y_index / self.height) * h)
                img[y][x] = img[y][x] + 1

        return img

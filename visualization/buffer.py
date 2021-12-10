import numpy as np


class AbstractBuffer:
    """ Abstract representation of a buffer """

    def __init__(self, details: dict):
        """ Initialize buffer using detail information from BSON """
        self.name = details["name"]
        self.accesses = []

    def add_memory_access(self, details: dict):
        """ Register a new memory access using detail information from BSON """
        pass

    def sanity_checks(self):
        """ Run basic consistency checks on buffer and memory access data """
        pass

    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        """ Calculate heatmap for given timerange.
        If one or more dimension is larger than max_res, memory accesses are mapped onto the interval [0, max_res)
        """
        pass


BufferCollection = dict[int, AbstractBuffer]


class Buffer1D(AbstractBuffer):
    class MemoryAccess:
        def __init__(self, details: dict):
            self.time = details["time"]
            self.index = details["index"]

    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]

    def add_memory_access(self, details: dict):
        self.accesses.append(Buffer1D.MemoryAccess(details))

    def sanity_checks(self):
        assert self.height >= 0
        assert self.name and self.name.strip()

        for access in self.accesses:
            assert 0 <= access.index < self.height
            assert 0 <= access.time

    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        start_time, end_time = timerange

        w = 1
        h = min(self.height, max_res)
        shape = (h, w)

        img = np.zeros(shape=shape)

        for access in self.accesses:
            if start_time <= access.time <= end_time:
                x = 0
                y = int((access.index / self.height) * h)
                img[y][x] = img[y][x] + 1

        return img


class Buffer2D(AbstractBuffer):
    class MemoryAccess:
        def __init__(self, details: dict, pitch: int):
            self.time = details["time"]
            index = details["index"]

            # convert 1D index into 2D coordinates
            self.x = index // pitch
            self.y = index % pitch

    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]
        self.width = details["width"]
        self.pitch = details["pitch"]

    def add_memory_access(self, details: dict):
        self.accesses.append(Buffer2D.MemoryAccess(details, self.pitch))

    def sanity_checks(self):
        assert 0 <= self.height
        assert 0 <= self.width <= self.pitch
        assert self.name and self.name.strip()

        for access in self.accesses:
            assert 0 <= access.x < self.width
            assert 0 <= access.y < self.height
            assert 0 <= access.time

    # @functools.lru_cache(maxsize=100)
    def generate_heatmap(self, timerange, max_res) -> np.ndarray:
        start_time, end_time = timerange

        w = min(self.width, max_res)
        h = min(self.height, max_res)
        shape = (h, w)

        img = np.zeros(shape=shape)

        for access in self.accesses:
            if start_time <= access.time <= end_time:
                x = int((access.x / self.width) * w)
                y = int((access.y / self.height) * h)
                img[y][x] = img[y][x] + 1

        return img

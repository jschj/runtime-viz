class AbstractBuffer:
    """ Abstract representation of a buffer """
    def __init__(self, details: dict):
        """ Initialize buffer using detail information from BSON """
        self.name = details["name"]
        self.accesses = []

    def add_memory_access(self, details: dict):
        pass

    def sanity_checks(self):
        pass


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

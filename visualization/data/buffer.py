from enum import Enum


class AbstractBuffer:
    def __init__(self, details: dict):
        self.name = details["name"]


class Buffer1D(AbstractBuffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]


class Buffer2D(AbstractBuffer):
    def __init__(self, details: dict):
        super().__init__(details)
        self.height = details["height"]
        self.width = details["width"]
        self.pitch = details["pitch"]


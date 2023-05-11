import enum


class Action(enum.IntEnum):
    INTERPRETER = enum.auto()
    JIT = enum.auto()
    RAISE_EXCEPTION = enum.auto()

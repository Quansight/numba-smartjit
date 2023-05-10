"""
This file will be copied to a temporary directory in order to
exercise caching compiled Numba functions.

Copied from "cache_usecases.py" with the jit decorator replaced by @smart_jit

See test_dispatcher.py.
"""

import sys

import numpy as np
from numba.tests.support import TestCase

from smart_jit import smart_jit


@smart_jit(cache=True, nopython=True)
def simple_usecase(x):
    return x


def simple_usecase_caller(x):
    return simple_usecase(x)


@smart_jit(cache=True, nopython=True)
def add_usecase(x, y):
    return x + y + Z


@smart_jit(cache=True, forceobj=True)
def add_objmode_usecase(x, y):
    object()
    return x + y + Z


@smart_jit(nopython=True)
def add_nocache_usecase(x, y):
    return x + y + Z


@smart_jit(cache=True, nopython=True)
def inner(x, y):
    return x + y + Z


@smart_jit(cache=True, nopython=True)
def outer(x, y):
    return inner(-y, x)


@smart_jit(cache=False, nopython=True)
def outer_uncached(x, y):
    return inner(-y, x)


@smart_jit(cache=True, forceobj=True)
def looplifted(n):
    object()
    res = 0
    for i in range(n):
        res = res + i
    return res


@smart_jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 2


renamed_function1 = ambiguous_function


@smart_jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 6


renamed_function2 = ambiguous_function


def make_closure(x):
    @smart_jit(cache=True, nopython=True)
    def closure(y):
        return x + y

    return closure


closure1 = make_closure(3)
closure2 = make_closure(5)
closure3 = make_closure(7)
closure4 = make_closure(9)


biggie = np.arange(10**6)


@smart_jit(cache=True, nopython=True)
def use_big_array():
    return biggie


Z = 1


class _TestModule(TestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        self.assertPreciseEqual(mod.add_usecase(2, 3), 6)
        self.assertPreciseEqual(mod.add_objmode_usecase(2, 3), 6)
        self.assertPreciseEqual(mod.outer_uncached(3, 2), 2)
        self.assertPreciseEqual(mod.outer(3, 2), 2)


@smart_jit(cache=True)
def first_class_function_mul(x):
    return x * x


@smart_jit(cache=True)
def first_class_function_add(x):
    return x + x


@smart_jit(cache=True)
def first_class_function_usecase(f, x):
    return f(x)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)


@smart_jit(parallel=True, cache=True, nopython=True)
def parfor_usecase(ary):
    return ary * ary + ary

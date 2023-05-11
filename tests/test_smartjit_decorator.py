import warnings

import numpy as np
import pytest
from numba.core.event import EventStatus, Listener, install_listener

from smart_jit import Action, jit, smart_jit_events


class CustomListener(Listener):
    def __init__(self) -> None:
        self.triggered = False
        super().__init__()


class CompilerListener(CustomListener):
    def on_start(self, event):
        self.triggered = True
        assert event.status == EventStatus.START
        assert event.kind == smart_jit_events["jit"]

    def on_end(self, event):
        pass


class InterpreterListener(CustomListener):
    def on_start(self, event):
        self.triggered = True
        assert event.status == EventStatus.START
        assert event.kind == smart_jit_events["interpreter"]

    def on_end(self, event):
        pass


class ListenerNotTriggeredException(Exception):
    ...


def add(a, b):
    return a + b


def sum_fast(A):
    acc = 0.0
    # with fastmath, the reduction can be vectorized as floating point
    # reassociation is permitted.
    for x in A:
        acc += np.sqrt(x)
    return acc


def use_jit_sum_fast(A):
    # for small arrays, just interpret
    if len(A) > 1_000:
        return Action.JIT
    return Action.INTERPRETER


def enable_jit(*args, **kwargs):
    return Action.JIT


def use_interpreter(*args, **kwargs):
    return Action.INTERPRETER


def check_listener(kind, listener, cfunc, args):
    with install_listener(kind, listener):
        cfunc(*args)
    # ensure listener was triggered
    if not listener.triggered:
        raise ListenerNotTriggeredException()


def check_interpreter_listener(cfunc, args):
    listener = InterpreterListener()
    return check_listener(
        smart_jit_events["interpreter"], listener, cfunc, args
    )


def check_compiler_listener(cfunc, args):
    listener = CompilerListener()
    return check_listener(smart_jit_events["jit"], listener, cfunc, args)


@pytest.mark.parametrize("use_jit", [enable_jit, use_interpreter])
def test_use_jit(use_jit):
    cfunc = jit(use_jit=use_jit)(add)
    if use_jit in (False, use_interpreter):
        check_interpreter_listener(cfunc, (2, 3))
    else:
        check_compiler_listener(cfunc, (2, 3))


@pytest.mark.parametrize("A", [np.arange(10), np.arange(100_000)])
def test_use_jit_sum_fast(A):
    cfunc = jit(use_jit=use_jit_sum_fast)(sum_fast)
    if len(A) == 10:
        check_interpreter_listener(cfunc, (A,))
    else:
        check_compiler_listener(cfunc, (A,))


def test_interpreter_action():
    _jit = jit("int64(int64, int64)", use_jit=use_interpreter)
    cfunc = _jit(add)
    check_interpreter_listener(cfunc, ("Hello, ", "World"))

    # Compiling should fail
    with pytest.raises(ListenerNotTriggeredException):
        check_compiler_listener(cfunc, ("Hello, ", "World"))


@pytest.mark.parametrize("inp", [(2, 3), (2.2, 4.4)])
def test_interpreter_action_with_signature(inp):
    _jit = jit("float64(float64, float64)", use_jit=use_interpreter)
    cfunc = _jit(add)
    check_compiler_listener(cfunc, inp)

    # Interpreter Listener should fail
    with pytest.raises(ListenerNotTriggeredException):
        check_interpreter_listener(cfunc, inp)


def test_actions_combined():
    def use_jit_func(a, b):
        if isinstance(a, int) and isinstance(b, int):
            # JIT compile for integers
            return Action.JIT
        elif isinstance(a, float) and isinstance(b, float):
            # Interpreter for floats
            return Action.INTERPRETER
        else:
            # raise exception otherwise
            return Action.RAISE_EXCEPTION

    _jit = jit(use_jit=use_jit_func)
    cfunc = _jit(add)

    # calling cfunc with integers or floats works
    check_compiler_listener(cfunc, (2, 4))
    check_interpreter_listener(cfunc, (2.2, 4.4))

    with pytest.raises(TypeError):
        check_compiler_listener(cfunc, ("Hello, ", "World"))


def test_actions_combined_with_jit_signature():
    def use_jit_func(a, b):
        if isinstance(a, int) and isinstance(b, int):
            # This will not happen as a jit function for integers
            # was previously compiled. Raising a RuntimeError to check if that
            # is the case
            raise RuntimeError()
        elif isinstance(a, float) and isinstance(b, float):
            # Run function in the interpreter
            return Action.INTERPRETER
        else:
            # raise exception for other types
            return Action.RAISE_EXCEPTION

    _jit = jit("int64(int64, int64)", use_jit=use_jit_func)
    cfunc = _jit(add)

    # calling cfunc with integers or floats works
    check_compiler_listener(cfunc, (2, 4))
    check_interpreter_listener(cfunc, (2.2, 4.4))

    with pytest.raises(TypeError):
        check_compiler_listener(cfunc, ("Hello, ", "World"))


def test_invalid_call():
    cfunc = jit("int64(int64, int64)", warn_on_fallback=True)(add)
    with pytest.raises(TypeError):
        cfunc("Hello", "world")


def test_no_warn_on_fallback():
    cfunc = jit("float64(float64, float64)", warn_on_fallback=True)(add)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfunc(2, 3)
        cfunc(2.2, 3.3)


_options = [
    dict(nopython=True),
    dict(nopython=True, fastmath=True),
    dict(nopython=True, fastmath=True, use_jit=enable_jit),
    dict(nopython=True, fastmath=True, use_jit=use_interpreter),
]


@pytest.mark.parametrize("options", _options)
def test_smart_jit_with_options(options):
    _jit = jit(**options)
    cfunc = _jit(add)
    assert cfunc(2, 3) == 5
    assert cfunc(2.2, 3.3) == 5.5
    assert cfunc("hello", ", world") == "hello, world"


def test_invalid_use_jit_type():
    with pytest.raises(TypeError):
        jit(use_jit="always")(add)

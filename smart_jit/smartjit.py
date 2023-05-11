"""
"""

__all__ = ["jit", "smart_jit_events"]


from typing import Callable

from numba import jit as numba_jit
from numba.core import event
from numba.core.registry import CPUDispatcher
from numba.core.target_extension import (
    CPU,
    dispatcher_registry,
    target_registry,
)

from .action import Action
from .warnings import NumbaInterpreterModeWarning

# 1. For jitted functions with a cache we can use a jitted function
# if it's in the cache and an interpreted function otherwise.
#
# 2. We could add a "dispatching" logic, an optional function to pass to the
# numba.jit decorator which will decide whether to use the jit or not.
# We currently do this manually in a number of places.

smart_jit_events = dict(
    jit="jit_execution", interpreter="interpreter_execution"
)


class SmartJitDispatcher(CPUDispatcher):
    def _default_checker(*args, **kwargs):
        return Action.JIT

    def __init__(self, *args, targetoptions, **kwargs):
        """
        use_jit: Callable
            Custom logic passed to the dispatcher object which will decide
            whether to use the jit or not. Default is to always use jit.

        warn_on_fallback: bool
            Set to True to warn when jit compilation/execution falls back
            to interpreter mode. Default value is False
        """
        self.use_jit = targetoptions.pop(
            "use_jit", SmartJitDispatcher._default_checker
        )
        self.warn_on_fallback = targetoptions.pop("warn_on_fallback", False)
        super().__init__(*args, targetoptions=targetoptions, **kwargs)

    def _emit_fallback_warning(self, *args):
        import warnings

        func_name = self.py_func.__name__
        arg_tys = [self.typeof_pyval(a) for a in args]
        msg = f"{func_name}({', '.join(map(str, arg_tys))}) not using JIT"
        if self.warn_on_fallback:
            warnings.warn(msg, NumbaInterpreterModeWarning)

    def _get_signature_from_args(self, *args, **kwargs):
        assert not kwargs, "kwargs not handled"
        args = tuple([self.typeof_pyval(a) for a in args])
        return args

    def _function_in_cache(self, *args, **kwargs):
        assert not kwargs, "kwargs not handled"
        sig_args = tuple([self.typeof_pyval(a) for a in args])

        # snippet copied from dispatcher.py::Dispatcher::compile
        cres = self._cache.load_overload(sig_args, self.targetctx)
        if cres is not None:
            self._cache_hits[sig_args] += 1
            # XXX fold this in add_overload()? (also see compiler.py)
            if not cres.objectmode:
                self.targetctx.insert_user_function(
                    cres.entry_point, cres.fndesc, [cres.library]
                )
            self.add_overload(cres)
            return cres.entry_point
        return None

    def _function_in_overload(self, *args, **kwargs):
        # partially copied from dispatcher.py::explain_ambiguous
        args = tuple([self.typeof_pyval(a) for a in args])
        sigs = self.nopython_signatures
        assert not kwargs, "kwargs not handled"
        func = self.typingctx.resolve_overload(
            self.py_func, sigs, args, kwargs, unsafe_casting=False
        )
        return True if func else False

    def can_compile(self):
        return self._can_compile

    def _fallback_interpreter(self, *args, **kwargs):
        # fallback to interpreter if cannot use jit or compilation is
        # disabled
        event.start_event(smart_jit_events["interpreter"])
        self._emit_fallback_warning(*args)
        ret = self.py_func(*args, **kwargs)
        event.end_event(smart_jit_events["interpreter"])
        return ret

    def _run_jit_func(self, *args, **kwargs):
        # function in cache
        event.start_event(smart_jit_events["jit"])
        ret = super().__call__(*args, **kwargs)
        event.end_event(smart_jit_events["jit"])
        return ret

    def __call__(self, *args, **kwargs):
        if self._function_in_overload(*args, **kwargs):
            return self._run_jit_func(*args, **kwargs)

        if self._function_in_cache(*args, **kwargs):
            return self._run_jit_func(*args, **kwargs)

        # Run use_jit function
        jit_action = self.use_jit(*args, **kwargs)

        if jit_action == Action.JIT:
            old_value = self._can_compile
            self._can_compile = True
            self._run_jit_func(*args, **kwargs)
            self._can_compile = old_value
        elif jit_action == Action.INTERPRETER:
            return self._fallback_interpreter(*args, **kwargs)
        elif jit_action == Action.RAISE_EXCEPTION:
            self._explain_matching_error(*args, **kwargs)
        else:
            msg = (
                'Invalid value returned from "use_jit" keyword. Expected '
                'one of "INTERPRETER, JIT_COMPILER, RAISE_EXCEPTION" '
                f'but got "{jit_action}"'
            )
            raise TypeError(msg)

        if jit_action and not self.can_compile():
            self._explain_matching_error(*args, **kwargs)

        if not jit_action:
            return self._fallback_interpreter(*args, **kwargs)

        return self._run_jit_func(*args, **kwargs)


class SmartJIT(CPU):
    ...


target_registry["SmartJitJIT"] = SmartJIT


dispatcher_registry[target_registry["SmartJitJIT"]] = SmartJitDispatcher


def jit(
    *args,
    use_jit=SmartJitDispatcher._default_checker,
    warn_on_fallback=False,
    **kws,
):
    """
    This decorator is used to compile a Python function into native code.
    custom options:
        use_jit: Callable
            Custom logic passed to the dispatcher object which will decide
            whether to use the jit or not. Default is to always use jit.

        warn_on_fallback: Callable or bool
            Set to True to warn when jit compilation/execution falls back
            to interpreter mode. Default value is False

    """
    if not isinstance(use_jit, Callable):
        msg = f"'use_jit' must be a Callable. Got {type(use_jit)}"
        raise TypeError(msg)

    kws["use_jit"] = use_jit
    kws["warn_on_fallback"] = warn_on_fallback
    return numba_jit(
        *args,
        **kws,
        _target="SmartJitJIT",
    )

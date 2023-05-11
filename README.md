# numba-smartjit

## Intro

smartjit `@jit` decorator adds extra customization of when code execution should fall back to the interpreter. It works as follow:

1. For jitted functions with cache (overloads), use the jitted function if available, and interpreted code otherwise
2. Add a **dispatching** logic, an optional function to pass to the jit decorator, which will decide wether to use jit or not.

## Changes

smartjit `@jit` decorator accepts the same set of argument as `@numba.jit` or `@numba.jit` with the addition of two new keyword arguments:

- `use_jit: Callback`
    - Callback function which returns an `smart_jit.Action`, determining whether to use jit compilation. `Action.INTERPRETER` will cause the function to always be interpreted, while `Action.JIT` will cause the function to always be JITted. If a callback function is passed, it will be evaluated on each function call, and the result will determine whether that call should be jitted or interpreted.
- `warn_on_fallback: bool`
    - Enabling this option will trigger a warning when JIT compilation/execution fails to utilize the JIT compiler and instead defaults to using the interpreter. This feature can be useful for debugging purposes. Default is `False`.

We also implement an Enum, named `Action`, which contains the set of possible actions one can return from `use_jit` callable:
- `Action.INTERPRETER`: Fallback execution to the interpreter
- `Action.JIT`: JIT compile and execute
- `Action.RAISE_EXCEPTION`: Raise no match `TypeError`

## How to use it

```python
from smart_jit import jit, Action
import numpy as np

def use_jit_sum_fast(A):
    # use jit compilation when length of A is greater than 100_000
    if len(A) > 100_000:
        return Action.JIT
    return Action.INTERPRETER

@jit(fastmath=True, use_jit=use_jit_sum_fast, warn_on_fallback=True)
def sum_fast(A):
    acc = 0.0
    # with fastmath, the reduction can be vectorized as floating point
    # reassociation is permitted.
    for x in A:
        acc += np.sqrt(x)
    return acc

A_small = np.arange(1_000, dtype=np.float64)
A_big = np.arange(1_000_000, dtype=np.float64)
```

```python
In [1]: sum_fast(A_small)  # interpreter
/Users/guilhermeleobas/git/numba-smartjit/smartjit.py:45: NumbaInterpreterModeWarning: sum_fast not using JIT
  warnings.warn(msg, NumbaInterpreterModeWarning)
Out[1]: 21065.833110879048

In [2]: sum_fast(A_big)  # will trigger jit compilation + execution
Out[2]: 666666166.4588218
```

In the example above, calling `sum_fast` with a `A_big` triggered jit compilation, whereas calling with `A_small` didn’t.

One important thing to notice is, after `sum_fast` is compiled for `A_big`, calling `sum_fast` again for `A_small` will now call the jitted version of `sum_fast`, since now there is an overload that matches the provided argument:

```python
In [3]: sum_fast.signatures
Out[3]: [(array(float64, 1d, C),)]

In [4]: sum_fast(A_small)
Out[4]: 21065.83311087906
```

### Providing signatures ahead-of-time

It is also possible to provide signatures ahead-of-time to the `@jit` decorator:

```python
from smart_jit import jit, Action

def use_jit(a, b):
    # fallback to interpreter mode
    return Action.INTERPRETER

@jit(['int64(int64, int64)', 'float64(float64, float64)'],
             use_jit=use_jit, warn_on_fallback=True)
def add(a, b):
    return a + b
```

```python
In [1]: add.signatures
Out[1]: [(int64, int64), (float64, float64)]

In [2]: add(2, 3)
Out[2]: 5

In [3]: add(2.2, 4.4)
Out[3]: 6.6000000000000005
```

Calling with a type that was not specified before will use the behavior returned by the `use_jit` function.

```python
In [4]: add('hello', ', world')
/Users/guilhermeleobas/git/numba-smartjit/smart_jit.py:62: NumbaInterpreterModeWarning: add(unicode_type, unicode_type) not using JIT
  warnings.warn(msg, NumbaInterpreterModeWarning)
Out[4]: 'hello, world'

In [5]: add.signatures
Out[5]: [(int64, int64), (float64, float64)]
```

This differs from other other decorators in Numba, which raises a `TypeError` when a matching error happens.

```python
from numba import njit
@njit('int32(int32, int32)')
def fn(a, b):
    return a
```

```python
In [1]: fn('hello', 'world')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[1], line 1
----> 1 fn('hello', 'world')

File ~/git/numba/numba/core/dispatcher.py:703, in _DispatcherBase._explain_matching_error(self, *args, **kws)
    700 args = [self.typeof_pyval(a) for a in args]
    701 msg = ("No matching definition for argument type(s) %s"
    702        % ', '.join(map(str, args)))
--> 703 raise TypeError(msg)

TypeError: No matching definition for argument type(s) unicode_type, unicode_type
```

### Raising exception on unexpected types

It is possible to raise an exception when `use_jit` is called with *unexpected* types. This can be achieved by returning `Action.RAISE_EXCEPTION` from the callback:

```python
from smart_jit import smart_jit, Action

def use_jit(a):
    if isinstance(a, int):
        return Action.JIT
    elif isinstance(a, str):
        return Action.RAISE_EXCEPTION
    else:
        return Action.INTERPRETER

@smart_jit(use_jit=use_jit, warn_on_fallback=True)
def double(a):
    return a + a
```

```python
In [1]: double(3)
Out[1]: 6

In [2]: double(4.4)
/Users/guilhermeleobas/git/numba-smartjit/smart_jit.py:62: NumbaInterpreterModeWarning: double(float64) not using JIT
  warnings.warn(msg, NumbaInterpreterModeWarning)
Out[2]: 8.8

In [3]: double.signatures
Out[3]: [(int64,)]

In [4]: double('hello')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[4], line 1
----> 1 double('hello')

File ~/git/numba-smartjit/smart_jit.py:133, in SmartJitDispatcher.__call__(self, *args, **kwargs)
    131     return self._fallback_interpreter(*args, **kwargs)
    132 elif jit_action == Action.RAISE_EXCEPTION:
--> 133     self._explain_matching_error(*args, **kwargs)
    134 else:
    135     msg = (
    136         'Invalid value returned from "use_jit" keyword. Expected '
    137         'one of "INTERPRETER, JIT_COMPILER, RAISE_EXCEPTION" '
    138         f'but got "{jit_action}"'
    139     )

File ~/git/numba/numba/core/dispatcher.py:703, in _DispatcherBase._explain_matching_error(self, *args, **kws)
    700 args = [self.typeof_pyval(a) for a in args]
    701 msg = ("No matching definition for argument type(s) %s"
    702        % ', '.join(map(str, args)))
--> 703 raise TypeError(msg)

TypeError: No matching definition for argument type(s) unicode_type
```

### smart_jit with caching enabled (`cache=True`)

If present, cached functions are loaded on demand. When executing a function, `smart_jit` will check if there is a function in cache that matches the signature before calling `use_jit`.

```python
from smart_jit import jit, Action

def use_jit(a):
    print(f'called "use_jit" with {a}')
    return Action.JIT

@jit(use_jit=use_jit, cache=True)
def incr(a):
    return a + 1
```

Calling for the first time will trigger JIT compilation and caching:
```python
$ ipython -i example.py

In [1]: incr(4)
called "use_jit" with <class 'int'>
Out[1]: 5
````

Calling the same function a second time will use the cached overload:
```python
$ ipython -i example.py

In [1]: incr(4)
Out[1]: 5

In [2]: # But only if the signature was previously cached

In [3]: incr(1.23)
called "use_jit" with <class 'float'>
Out[3]: 2.23
```


## Caveats

It is possible to track wether a function is using jit compilation/execution with the help of event listeners. Numba provides an API for listening to certain events that happens inside the compiler. For the `@smart_jit` work, I’ve implemented two new event kinds (`jit_execution` and `interpreter_execution`) that are notified when jit or interpreter execution happens. Example:

```python
from smart_jit import jit, Action
from numba.core import event

class CustomListener(event.Listener):
    def on_start(self, event):
        print(f'Start {event.kind}...')

    def on_end(self, event):
        print(f'End {event.kind}...')

def int_jit(a):
    if isinstance(a, int):
        return Action.JIT
    return Action.INTERPRETER

@jit(use_jit=int_jit)
def incr(a):
    return a + 1
```

```python
In [1]: listener = CustomListener()
   ...: with event.install_listener("jit_execution", listener):
   ...:     incr(4)
   ...:
Start jit_execution...
End jit_execution...
```

Calling `incr` with a float value will not trigger the `jit_execution` event, but will trigger `interpreter_execution`:

```python
In [2]: with event.install_listener("jit_execution", listener):
   ...:     incr(1.23)
   ...:

In [3]: with event.install_listener("interpreter_execution", listener):
   ...:     incr(1.23)
   ...:
Start interpreter_execution...
End interpreter_execution...
```

## Limitations

All limitations of Numba [`@jit`](https://numba.readthedocs.io/en/stable/user/jit.html) persist.
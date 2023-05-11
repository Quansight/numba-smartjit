# numba-smartjit

## Intro

smartjit `@jit` decorator adds extra customization of when code execution should fall back to the interpreter. It works as follow:

1. For jitted functions with cache (overloads), use the jitted function if available, and interpreted code otherwise
2. Add a **dispatching** logic, an optional function to pass to the jit decorator, which will decide wether to use jit or not.

## Install

numba-smartjit is available on PyPI and can be installed with the command below:

```bash
pip install numba-smartjit
```

## How to use it

[howto.md](howto.md)

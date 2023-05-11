from smart_jit import jit, Action


def use_jit(a):
    if isinstance(a, (int, float)):
        # jit compile for int and float
        return Action.JIT
    elif isinstance(a, str):
        # run string function in the interpreter
        return Action.INTERPRETER
    else:
        # RAISE_EXCEPTION will raise TypeError with no match message
        return Action.RAISE_EXCEPTION


@jit(use_jit=use_jit, warn_on_fallback=True)
def double(a):
    return a + a


# jit compile + execution
print(double(3))
print(double(4.4))
print(f"List of signatures: {double.signatures}")

# double(str) will run in the interpreter
print(double("Test"))
print(f"List of signatures: {double.signatures}")

# anything different from that will raise an exception
try:
    a_list = [1, 2, 3]
    print(double(a_list))
except TypeError:
    print("Exception happened")

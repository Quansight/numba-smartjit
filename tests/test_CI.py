import os
import re
import sys

import pytest


def parse_version(version):
    """Return parsed version tuple from version string.

    For instance:

    >>> parse_version('1.2.3dev4')
    (1, 2, 3, 'dev4')
    """

    m = re.match(r"(\d+)[.](\d+)[.](\d+)(.*)", version)
    if m is not None:
        major, minor, micro, dev = m.groups()
        if not dev:
            return (int(major), int(minor), int(micro))
        return (int(major), int(minor), int(micro), dev)

    m = re.match(r"(\d+)[.](\d+)(.*)", version)
    if m is not None:
        major, minor, dev = m.groups()
        if not dev:
            return (int(major), int(minor))
        return (int(major), int(minor), dev)

    m = re.match(r"(\d+)(.*)", version)
    if m is not None:
        major, dev = m.groups()
        if not dev:
            return (int(major),)
        return (int(major), dev)

    if version:
        return (version,)
    return ()


def test_python_version():
    varname = "EXPECTED_PYTHON_VERSION"
    current = tuple(sys.version_info)
    expected = os.environ.get(varname)
    if os.environ.get("CI"):
        assert expected is not None, (varname, current)
    if expected is None:
        msg = (
            f"Undefined environment variable {varname}, "
            "cannot test python version "
            f'(current={".".join(map(str, current))})'
        )
        pytest.skip(msg)
    expected = parse_version(expected)
    current_stripped = current[: len(expected)]
    assert expected == current_stripped


def test_numba_version():
    varname = "EXPECTED_NUMBA_VERSION"
    import numba

    current = parse_version(numba.__version__)
    expected = os.environ.get(varname)
    if expected is None:
        msg = (
            f"Undefined environment variable {varname}, "
            "cannot test numba version "
            f'(current={".".join(map(str, current))})'
        )
        pytest.skip(msg)
    expected = parse_version(expected)
    current_stripped = current[: len(expected)]
    assert expected == current_stripped

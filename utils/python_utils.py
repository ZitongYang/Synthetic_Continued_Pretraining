"""Credit to Alpaca project and Xuechen Li (lxuechen)."""

import io
import pathlib
from typing import Callable, Optional, Sequence, Union


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def read(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    content = f.read()
    if strip:
        content = content.strip()
    f.close()
    return content

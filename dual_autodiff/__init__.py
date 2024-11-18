# dual_autdiff/__init__.py

# Import the main Company class for easy access
from .dual import Dual

# Import version information for the package
from .version import __version__ 

from .dual import Dual
from .autodiff_tools import (
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    exp,
    log,
    sqrt,
    pow,
    auto_diff,
)

__all__ = [
    "Dual",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "pow",
    "auto_diff",
]
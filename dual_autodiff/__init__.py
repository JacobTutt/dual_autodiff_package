# dual_autdiff/__init__.py

# Import the main Company class for easy access
from .dual import *

# Import version information for the package
from .version import __version__ 

from .dual import Dual
from .autodiff_tools import *

__all__ = [
    "Dual",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "pow",
    "auto_diff",
    "multi_auto_diff",
]
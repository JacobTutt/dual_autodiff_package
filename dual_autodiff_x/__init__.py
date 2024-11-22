# Import the main Dual class from the compiled extension
from .dual import *

# Import version information for the package
from dual_autodiff.version import __version__

# Import mathematical functions from the compiled extension
from .autodiff_tools import *

# Define the public API of the package
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
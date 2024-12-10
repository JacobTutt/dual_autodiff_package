# Import the main Dual class explicitly
from dual_autodiff_x.dual import Dual

# Import specific functions from autodiff_tools
from dual_autodiff_x.autodiff_tools import (
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    sinh,
    cosh,
    tanh,
    exp,
    log,
    sqrt,
    pow,
    auto_diff,
    multi_auto_diff,
)

# Import version information from the main package
from dual_autodiff.version import __version__

# Define the public API of the package explicitly
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
    "__version__",
]
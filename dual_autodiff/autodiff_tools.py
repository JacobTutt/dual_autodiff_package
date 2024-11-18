from .dual import Dual
from math import (
    sin as math_sin,
    cos as math_cos,
    tan as math_tan,
    asin as math_asin,
    acos as math_acos,
    atan as math_atan,
    sinh as math_sinh,
    cosh as math_cosh,
    tanh as math_tanh,
    exp as math_exp,
    log as math_log,
    sqrt as math_sqrt,
    pow as math_pow,
)


# This file integrated the class's mathema functions with the math module's functions
# Allows more seemless integration of the Dual class with the math module

# Call Dual's sin() method
def sin(x):
    """
    Compute the sine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The sine of the input.

    Examples:
        >>> from dual_autodiff import sin, Dual
        >>> sin(0)
        0.0
        >>> sin(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    """
    if isinstance(x, Dual):
        return x.sin()
    return math_sin(x)

# Call Dual's cos() method
def cos(x):
    """
    Compute the cosine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The cosine of the input.

    Examples:
        >>> from dual_autodiff import cos, Dual
        >>> cos(0)
        1.0
        >>> cos(Dual(0, 1))
        Dual(real=1.0, dual=0.0)
    """
    if isinstance(x, Dual):
        return x.cos()
    return math_cos(x)

# Call Dual's tan() method
def tan(x):
    """
    Compute the tangent of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The tangent of the input.

    Raises:
        ValueError: The tangent function is undefined as cosine of real part equals 0

    Examples:
        >>> from dual_autodiff import tan, Dual
        >>> tan(0)
        0.0
        >>> tan(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    """
    if isinstance(x, Dual):
        return x.tan()
    return math_tan(x)

# Call Dual's arcsin() method

def asin(x):
    """
    Compute the arcsine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The arcsine of the input.
    
    Raises:
        ValueError: If the real part is outside the range [-1, 1].

    Examples:
        >>> from dual_autodiff import asin, Dual
        >>> asin(0)
        0.0
        >>> asin(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    """
    if isinstance(x, Dual):
        return x.asin()
    return math_asin(x)

# Call Dual's arccos() method
def acos(x):
    """
    Compute the arccosine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The arccosine of the input.

    Raises:
        ValueError: If the real part is outside the range [-1, 1].

    Examples:
        >>> from dual_autodiff import acos, Dual
        >>> acos(1)
        0.0
        >>> acos(Dual(1, 0))
        Dual(real=0.0, dual=0.0)
    """
    if isinstance(x, Dual):
        return x.acos()
    return math_acos(x)

# Call Dual's arctan() method
def atan(x):
    """
    Compute the arctangent of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The arctangent of the input.

    Examples:
        >>> from dual_autodiff import atan, Dual
        >>> atan(1)
        0.7853981633974483
        >>> atan(Dual(1, 1))
        Dual(real=0.7853981633974483, dual=0.5)
    """
    if isinstance(x, Dual):
        return x.atan()
    return math_atan(x)

# Call Dual's sinh() method
def sinh(x):
    """
    Compute the hyperbolic sine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The hyperbolic sine of the input.

    Examples:
        >>> from dual_autodiff import sinh, Dual
        >>> sinh(1)
        1.1752011936438014
        >>> sinh(Dual(1, 1))
        Dual(real=1.1752011936438014, dual=1.5430806348152437)
    """
    if isinstance(x, Dual):
        return x.sinh()
    return math_sinh(x)


# Call Dual's cosh() method

def cosh(x):
    """
    Compute the hyperbolic cosine of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The hyperbolic cosine of the input.

    Examples:
        >>> from dual_autodiff import cosh, Dual
        >>> cosh(1)
        1.5430806348152437
        >>> cosh(Dual(1, 1))
        Dual(real=1.5430806348152437, dual=1.1752011936438014)
    """
    if isinstance(x, Dual):
        return x.cosh()
    return math_cosh(x)

# Call Dual's tanh() method
def tanh(x):
    """
    Compute the hyperbolic tangent of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The hyperbolic tangent of the input.

    Examples:
        >>> from dual_autodiff import tanh, Dual
        >>> tanh(1)
        0.7615941559557649
        >>> tanh(Dual(1, 1))
        Dual(real=0.7615941559557649, dual=0.41997434161402614)
    """
    if isinstance(x, Dual):
        return x.tanh()
    return math_tanh(x)

# Call Dual's exp() method


def exp(x):
    """
    Compute the exponential of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The exponential of the input.

    Examples:
        >>> from dual_autodiff import exp, Dual
        >>> exp(1)
        2.718281828459045
        >>> exp(Dual(1, 1))
        Dual(real=2.718281828459045, dual=2.718281828459045)
    """
    if isinstance(x, Dual):
        return x.exp()
    return math_exp(x)

# Call Dual's log() method
def log(x):
    """
    Compute the natural logarithm of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The natural logarithm of the input.

    Raises:
        ValueError: If the real part of dual number is non-positive

    Examples:
        >>> from dual_autodiff import log, Dual
        >>> log(2)
        0.6931471805599453
        >>> log(Dual(2, 1))
        Dual(real=0.6931471805599453, dual=0.5)
    """
    if isinstance(x, Dual):
        return x.log()
    return math_log(x)

# Call Dual's sqrt() method

def sqrt(x):
    """
    Compute the square root of a number or a Dual number.

    Parameters:
        x (float or Dual): The input value.

    Returns:
        float or Dual: The square root of the input.

    Raises:
        ValueError: If the real part of dual number is negative.

    Examples:
        >>> from dual_autodiff import sqrt, Dual
        >>> sqrt(4)
        2.0
        >>> sqrt(Dual(4, 1))
        Dual(real=2.0, dual=0.25)
    """
    if isinstance(x, Dual):
        return x.sqrt()
    return math_sqrt(x)

# Call Dual's pow() method
def pow(x, n):
    """
    Compute a number or a Dual number raised to a power.

    Parameters:
        x (float or Dual): The base.
        n (float): The exponent.

    Returns:
        float or Dual: The result of raising `x` to the power `n`.

    Raises:
        TypeError: If n is not an int or float.

    Examples:
        >>> from dual_autodiff import pow, Dual
        >>> pow(2, 3)
        8
        >>> pow(Dual(2, 1), 3)
        Dual(real=8, dual=12.0)
    """
    if isinstance(x, Dual):
        return x.pow(n)
    return math_pow(x, n)

# Evaluates a function on a dual number and returns the dual part of result
# Corrosponds to derivative - ie preforms automatic differentiation
def auto_diff(func, x):
    """
    Evaluates the derivative of a function f at x using Dual number: x + ε.

    Parameters:
        func (callable): The function to differentiate.
        x (float): The point where the derivative is evaluated.

    Returns:
        float: The derivative of `f` at `x`.

    Raises:
        TypeError: If f is not callable
        TypeError: If input x a float, or int.

    Examples:
        >>> from dual_autodiff import auto_diff
        >>> auto_diff(lambda x: x**2 + x, 2)
        5.0
    """
    # Validate that f is callable function 
    if not callable(func):
        raise TypeError(f"function must be a callable function, got {type(func).__name__}.")

    # Validate that input x is a or scalar (float/ int)
    if not isinstance(x, (Dual, float, int)):
        raise TypeError(f"x must be a scalar number (float/int), got {type(x).__name__}.")
    
    value = func(Dual(x, 1))

    # This accounts for the case in which the function is constant and therefore resturns a constant non dual number
    # Assumes this is the case and creates a dual number with derivative 0
    if not isinstance(value, Dual):
        value = Dual(value, 0)
        
    return value.dual
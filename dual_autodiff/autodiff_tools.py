from .dual import Dual
import numpy as np
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
    Compute the sin of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The sine of the input.


    Examples:
        >>> from dual_autodiff import sin, Dual
    1. With a scalar input:
        >>> sin(0)
        0.0
    2. With a Dual number input:
        >>> sin(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    3. With a Dual numpy array input:
        >>> sin(np.array([Dual(2,3), Dual(3,4)]))
        array([Dual(real=0.9093..., dual=-1.2484...),
        Dual(real=0.1411..., dual=-3.9600...)],
        dtype=object)
    """

    # If input is a Dual number, call the sin() method from Dual class
    if isinstance(x, Dual):
        return x.sin()
    
    # If input is a numpy array, call the np.sin method which will call the Dual class's sin method
    if isinstance(x, np.ndarray):
        return np.sin(x)

    # If input is a scalar, call the math.sin method
    return math_sin(x)


# Call Dual's cos() method
def cos(x):
    """
    Compute the cos of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The cosine of the input.

    Examples:
        >>> from dual_autodiff import cos, Dual
    1. With a scalar input:
        >>> cos(0)
        1.0
    2. With a Dual number input:
        >>> cos(Dual(0, 1))
        Dual(real=1.0, dual=0.0)
    3. With a Dual numpy array input:
        >>> cos(np.array([Dual(2,3), Dual(3,4)]))
        array([Dual(real=-0.4161..., dual=-2.7278...),
        Dual(real=-0.9899..., dual=-0.1411...)],
        dtype=object)
    """

    # If input is a Dual number, call the cos() method from Dual class
    if isinstance(x, Dual):
        return x.cos()
    
    # If input is a numpy array, call the np.cos method which will call the Dual class's cos method
    if isinstance(x, np.ndarray):
        return np.cos(x)
    
    # If input is a scalar, call the math.cos method
    return math_cos(x)


# Call Dual's tan() method
# Errors thrown by Dual class (dual.py) automatically
def tan(x):
    """
    Compute the tangent of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The tangent of the input.

    Raises:
        ValueError: The tangent function is undefined as cosine of real part equals 0.

    Examples:
        >>> from dual_autodiff import tan, Dual
    1. With a scalar input:
        >>> tan(0)
        0.0
    2. With a Dual number input:
        >>> tan(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    3. With a Dual numpy array input:
        >>> tan(np.array([Dual(2,3), Dual(3,4)]))
        array([Dual(real=-2.1850..., dual=5.7744...),
        Dual(real=-0.1425..., dual=16.2574...)],
        dtype=object)
    """

    # If input is a Dual number, call the tan() method from Dual class
    if isinstance(x, Dual):
        return x.tan()
    
    # If input is a numpy array, call the np.tan method which will call the Dual class's tan method
    if isinstance(x, np.ndarray):
        return np.tan(x)
    
    # If input is a scalar, call the math.tan method
    return math_tan(x)


# Call Dual's arcsin() method
# Errors thrown by Dual class (dual.py) automatically
def arcsin(x):
    """
    Compute the arcsine of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The arcsine of the input.
    
    Raises:
        ValueError: If a real part is outside the range [-1, 1].

    Examples:
        >>> from dual_autodiff import arcsin, Dual
    1. With a scalar input:
        >>> arcsin(0)
        0.0
    2. With a Dual number input:
        >>> arcsin(Dual(0, 1))
        Dual(real=0.0, dual=1.0)
    3. With a Dual numpy array input:
        >>> arcsin(np.array([Dual(0.5,1), Dual(0.3,2)]))
        array([Dual(real=0.5235..., dual=1.1547...),
        Dual(real=0.3046..., dual=2.0910...)],
        dtype=object)
    """

    # If input is a Dual number, call the arcsin() method from Dual class
    if isinstance(x, Dual):
        return x.arcsin()
    
    # If input is a numpy array, call the np.arcsin method which will call the Dual class's arcsin method
    if isinstance(x, np.ndarray):
        return np.arcsin(x)
    
    # If input is a scalar, call the math.asin method
    return math_asin(x)


# Call Dual's arccos() method
# Errors thrown by Dual class (dual.py) automatically
def arccos(x):
    """
    Compute the arccosine of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Raises:
        ValueError: If the real part is outside the range [-1, 1].

    Examples:
        >>> from dual_autodiff import arccos, Dual
    1. With a scalar input:
        >>> arccos(1)
        0.0
    2. With a Dual number input:
        >>> arccos(Dual(1, 1))
        Dual(real=0.0, dual=-1.0)
    3. With a Dual numpy array input:
        >>> arccos(np.array([Dual(0.5,1), Dual(0.3,2)]))
        array([Dual(real=1.0471..., dual=-1.1547...),
        Dual(real=1.2661..., dual=-2.0910...)],
        dtype=object)
    """

    # If input is a Dual number, call the acos() method from Dual class
    if isinstance(x, Dual):
        return x.arccos()
    
    # If input is a numpy array, call the np.arccos method which will call the Dual class's acos method
    if isinstance(x, np.ndarray):
        return np.arccos(x)
    
    # If input is a scalar, call the math.acos method
    return math_acos(x)


# Call Dual's arctan() method
def arctan(x):
    """
    Compute the arctangent of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The arctangent of the input.

    Examples:
        >>> from dual_autodiff import arctan, Dual
    1. With a scalar input:
        >>> arctan(1)
        0.7853981633974483
    2. With a Dual number input:
        >>> arctan(Dual(1, 1))
        Dual(real=0.7853981633974483, dual=0.5)
    3. With a Dual numpy array input:
        >>> arctan(np.array([Dual(1,1), Dual(0.5,2)]))
        array([Dual(real=0.7853..., dual=0.5),
        Dual(real=0.4636..., dual=1.6)],
        dtype=object)
    """


    # If input is a Dual number, call the atan() method from Dual class
    if isinstance(x, Dual):
        return x.arctan()
    
    # If input is a numpy array, call the np.arctan method which will call the Dual class's atan method
    if isinstance(x, np.ndarray):
        return np.arctan(x)
    
    # If input is a scalar, call the math.atan method
    return math_atan(x)


# Call Dual's sinh() method
def sinh(x):
    """
    Compute the hyperbolic sine of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The hyperbolic sine of the input.

    Examples:
        >>> from dual_autodiff import sinh, Dual
    1. With a scalar input:
        >>> sinh(1)
        1.1752011936438014
    2. With a Dual number input:
        >>> sinh(Dual(1, 1))
        Dual(real=1.1752011936438014, dual=1.5430806348152437)
    3. With a Dual numpy array input:
        >>> sinh(np.array([Dual(1,1), Dual(0.5,2)]))
        array([Dual(real=1.1752..., dual=1.5430...),
        Dual(real=0.5210..., dual=2.1276...)],
        dtype=object)
    """


    # If input is a Dual number, call the sinh() method from Dual class
    if isinstance(x, Dual):
        return x.sinh()
    
    # If input is a numpy array, call the np.sinh method which will call the Dual class's sinh method
    if isinstance(x, np.ndarray):
        return np.sinh(x)
    
    # If input is a scalar, call the math.sinh method
    return math_sinh(x)


# Call Dual's cosh() method
def cosh(x):
    """
    Compute the hyperbolic cosine of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The hyperbolic cosine of the input.

    Examples:
        >>> from dual_autodiff import cosh, Dual
    1. With a scalar input:
        >>> cosh(1)
        1.5430806348152437
    2. With a Dual number input:
        >>> cosh(Dual(1, 1))
        Dual(real=1.5430806348152437, dual=1.1752011936438014)
    3. With a Dual numpy array input:
        >>> cosh(np.array([Dual(1,1), Dual(0.5,2)]))
        array([Dual(real=1.5430..., dual=1.1752...),
        Dual(real=1.1276..., dual=2.5210...)],
        dtype=object)
    """

    # If input is a Dual number, call the cosh() method from Dual class
    if isinstance(x, Dual):
        return x.cosh()
    
    # If input is a numpy array, call the np.cosh method which will call the Dual class's cosh method
    if isinstance(x, np.ndarray):
        return np.cosh(x)
    
    # If input is a scalar, call the math.cosh method
    return math_cosh(x)


# Call Dual's tanh() method
def tanh(x):
    """
    Compute the hyperbolic tangent of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The hyperbolic tangent of the input.

    Examples:
        >>> from dual_autodiff import tanh, Dual
    1. With a scalar input:
        >>> tanh(1)
        0.7615941559557649
    2. With a Dual number input:
        >>> tanh(Dual(1, 1))
        Dual(real=0.7615941559557649, dual=0.41997434161402614)
    3. With a Dual numpy array input:
        >>> tanh(np.array([Dual(1,1), Dual(0.5,2)]))
        array([Dual(real=0.7615..., dual=0.4199...),
        Dual(real=0.4621..., dual=1.7864...)],
        dtype=object)
    """

    # If input is a Dual number, call the tanh() method from Dual class
    if isinstance(x, Dual):
        return x.tanh()
    
    # If input is a numpy array, call the np.tanh method which will call the Dual class's tanh method
    if isinstance(x, np.ndarray):
        return np.tanh(x)
    
    # If input is a scalar, call the math.tanh method
    return math_tanh(x)


# Call Dual's exp() method
def exp(x):
    """
    Compute the exponential of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The exponential of the input.

    Examples:
        >>> from dual_autodiff import exp, Dual
    1. With a scalar input:
        >>> exp(1)
        2.718281828459045
    2. With a Dual number input:
        >>> exp(Dual(1, 1))
        Dual(real=2.718281828459045, dual=2.718281828459045)
    3. With a Dual numpy array input:
        >>> exp(np.array([Dual(1,1), Dual(0.5,2)]))
        array([Dual(real=2.7182..., dual=2.7182...),
        Dual(real=1.6487..., dual=3.2974...)],
        dtype=object)
    """

    # If input is a Dual number, call the exp() method from Dual class
    if isinstance(x, Dual):
        return x.exp()
    
    # If input is a numpy array, call the np.exp method which will call the Dual class's exp method
    if isinstance(x, np.ndarray):
        return np.exp(x)
    
    # If input is a scalar, call the math.exp method
    return math_exp(x)

# Call Dual's log() method
# Errors thrown by Dual class (dual.py) automatically
def log(x):
    """
    Compute the natural logarithm of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The natural logarithm of the input.

    Raises:
        ValueError: If a real part of dual number is non-positive

    Examples:
        >>> from dual_autodiff import log, Dual
    1. With a scalar input:
        >>> log(1)
        0.0
    2. With a Dual number input:
        >>> log(Dual(1, 1))
        Dual(real=0.0, dual=1.0)
    3. With a Dual numpy array input:
        >>> log(np.array([Dual(1,1), Dual(2,2)]))
        array([Dual(real=0.0, dual=1.0),
        Dual(real=0.6931..., dual=1.0)],
        dtype=object)
    """

    # If input is a Dual number, call the log() method from Dual class
    if isinstance(x, Dual):
        return x.log()
    
    # If input is a numpy array, call the np.log method which will call the Dual class's log method
    if isinstance(x, np.ndarray):
        return np.log(x)
    
    # If input is a scalar, call the math.log method
    return math_log(x)


# Call Dual's sqrt() method
# Errors thrown by Dual class (dual.py) automatically
def sqrt(x):
    """
    Compute the square root of a number, a Dual number, or a numpy array of Dual numbers.

    Parameters:
        x (float, Dual, or numpy.ndarray): The input value/values.

    Returns:
        float, Dual, or numpy.ndarray: The square root of the input.

    Raises:
        ValueError: If the real part of dual number is negative.

    Examples:
        >>> from dual_autodiff import sqrt, Dual
    1. With a scalar input:
        >>> sqrt(4)
        2.0
    2. With a Dual number input:
        >>> sqrt(Dual(4, 1))
        Dual(real=2.0, dual=0.25)
    3. With a Dual numpy array input:
        >>> sqrt(np.array([Dual(4,1), Dual(9,2)]))
        array([Dual(real=2.0, dual=0.25),
        Dual(real=3.0, dual=0.3333...)],
        dtype=object)
    """

    # If input is a Dual number, call the sqrt() method from Dual class
    if isinstance(x, Dual):
        return x.sqrt()
    
    # If input is a numpy array, call the np.sqrt method which will call the Dual class's sqrt method
    if isinstance(x, np.ndarray):
        return np.sqrt(x)
    
    # If input is a scalar, call the math.sqrt method
    return math_sqrt(x)


# Call Dual's pow() method
# Errors thrown by Dual class (dual.py) automatically
def pow(x, n):
    """
    Compute a number, a Dual number, or a numpy array of Dual numbers raised to a power.

    Parameters:
        x (float, Dual, or numpy.ndarray): The base.
        n (float): The exponent.

    Returns:
        float, Dual, or numpy.ndarray: The result of raising `x` to the power `n`.

    Raises:
        TypeError: If n is not an int or float.

    Examples:
        >>> from dual_autodiff import pow, Dual
    1. With a scalar input:
        >>> pow(2, 3)
        8
    2. With a Dual number input:
        >>> pow(Dual(2, 1), 3)
        Dual(real=8, dual=12.0)
    3. With a Dual numpy array input:
        >>> pow(np.array([Dual(2,1), Dual(3,2)]), 3)
        array([Dual(real=8, dual=12.0),
        Dual(real=27, dual=54.0)],
        dtype=object)
    """

    # If input is a Dual number, call the pow() method from Dual class
    if isinstance(x, Dual):
        return x.pow(n)
    
    # If input is a numpy array, call the np.power method which will call the Dual class's pow method
    if isinstance(x, np.ndarray):
        return np.power(x, n)
    
    # If input is a scalar, call the math.pow method
    return math_pow(x, n)


# Evaluates a function on a dual number and returns the dual part of result
# Corresponds to derivative - ie performs automatic differentiation
def auto_diff(func, x):
    """
    Evaluates the derivative of a function f at x using Dual number: x + ε.

    Parameters:
        func (callable): The function to differentiate.
        x (float, int, or numpy.ndarray): The point(s) where the derivative is evaluated.

    Returns:
        tuple: A tuple containing the real and dual parts of the result.
               If x is a scalar, returns (real, dual).
               If x is a numpy array, returns two numpy arrays (real_array, dual_array).

    Raises:
        TypeError: If func is not callable.
        TypeError: If input x is not a float, int, or numpy.ndarray containing scalar values.

    Examples:
        >>> from dual_autodiff import auto_diff
    1. With a scalar input:
        >>> auto_diff(lambda x: x**3 + 2*x**2 + x, 2)
        (14.0, 17.0)
    2. With a numpy array input:
        >>> auto_diff(lambda x: x**3 + 2*x**2 + x, np.array([1, 2, 3]))
        (array([ 4.0, 14.0, 34.0]), array([ 4.0, 17.0, 40.0]))
    """
    # Check that func is callable
    if not callable(func):
        raise TypeError(f"func must be a callable function, got {type(func).__name__}.")

    # Check that input x is a scalar (float/int) or numpy array
    if not isinstance(x, (float, int, np.ndarray)):
        raise TypeError(f"x must be a scalar number (float/int) or numpy.ndarray, got {type(x).__name__}.")

    if isinstance(x, (float, int)):
        # If x is a scalar, evaluate the function directly
        value = func(Dual(x, 1))
        # This accounts for the case in which the function is constant and therefore returns a constant non-dual number
        # Assumes this is the case and creates a dual number with derivative 0
        if not isinstance(value, Dual):
            value = Dual(value, 0)

    else:
        # Checks that is x is a numpy array, it contains scalar values
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("numpy.ndarray must contain scalar values (float/int).")
        
        # Compute the real and dual parts
        # As above accounts for the case in which the function is constant and therefore returns a constant non-dual number
        real_parts = [func(Dual(float(xi), 1)).real if isinstance(func(Dual(float(xi), 1)), Dual) else func(Dual(float(xi), 1)) for xi in x]
        dual_parts = [func(Dual(float(xi), 1)).dual if isinstance(func(Dual(float(xi), 1)), Dual) else 0 for xi in x]
        
        return np.array(real_parts), np.array(dual_parts)

    return value.real, value.dual

# Evaluates multiple functions value and derivative at x using Dual number/ automatic differentiation
# Input is a list of functions and a point x/ array of points
# Returns a list of tuples, each containing the real and dual parts of the result for each function
# uses auto_diff to evaluate each function
def multi_auto_diff(funcs, x):
    """
    Evaluates the derivatives of multiple functions at x using Dual number: x + ε.

    Parameters:
        funcs (list of callables): The functions to differentiate.
        x (float, int, or numpy.ndarray): The point(s) where the derivatives are evaluated.

    Returns:
        list of tuples: A list of tuples, each containing the real and dual parts of the result for each function.
                        If x is a scalar, each tuple is (real, dual).
                        If x is a numpy array, each tuple contains two numpy arrays (real_array, dual_array).

    Raises:
        TypeError: If any func in funcs is not callable.
        TypeError: If input x is not a float, int, or numpy.ndarray containing scalar values.

    Examples:
        >>> from dual_autodiff import multi_auto_diff
    1. With a scalar input:
        >>> multi_auto_diff([lambda x: x**3, lambda x: x**2], 2)
        [(8.0, 12.0), (4.0, 4.0)]
    2. With a numpy array input:
        >>> multi_auto_diff([lambda x: x**3, lambda x: x**2], np.array([1, 2, 3]))
        [(array([ 1.0,  8.0, 27.0]), array([ 3.0, 12.0, 27.0])), (array([1.0, 4.0, 9.0]), array([2.0, 4.0, 6.0]))]
    """
    results = []
    for func in funcs:
        if not callable(func):
            raise TypeError(f"Each func must be a callable function, got {type(func).__name__}.")
        results.append(auto_diff(func, x))

    # output structure: [[func1_f(x)], [func1_f'(x)], [func2_f(x)], [func2_f'(x)], ...]
    return results 
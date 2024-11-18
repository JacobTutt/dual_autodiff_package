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
    if isinstance(x, Dual):
        return x.sin() 
    return math_sin(x)

# Call Dual's cos() method
def cos(x):
    if isinstance(x, Dual):
        return x.cos()  
    return math_cos(x)

# Call Dual's tan() method
def tan(x):
    if isinstance(x, Dual):
        return x.tan() 
    return math_tan(x)

# Call Dual's arcsin() method
def asin(x):
    if isinstance(x, Dual):
        return x.arcsin()
    return math_asin(x)

# Call Dual's arccos() method
def acos(x):
    if isinstance(x, Dual):
        return x.arccos()  
    return math_acos(x)

# Call Dual's arctan() method
def atan(x):
    if isinstance(x, Dual):
        return x.arctan() 
    return math_atan(x)

# Call Dual's sinh() method
def sinh(x):
    if isinstance(x, Dual):
        return x.sinh()  
    return math_sinh(x)

# Call Dual's cosh() method
def cosh(x):
    if isinstance(x, Dual):
        return x.cosh() 
    return math_cosh(x)

# Call Dual's tanh() method
def tanh(x):
    if isinstance(x, Dual):
        return x.tanh() 
    return math_tanh(x)

# Call Dual's exp() method
def exp(x):
    if isinstance(x, Dual):
        return x.exp()  
    return math_exp(x)

# Call Dual's log() method
def log(x):
    if isinstance(x, Dual):
        return x.log() 
    return math_log(x)

# Call Dual's sqrt() method
def sqrt(x):
    if isinstance(x, Dual):
        return x.sqrt()  # Call Dual's sqrt() method
    return math_sqrt(x)

# Call Dual's pow() method
def pow(x, n):
    if isinstance(x, Dual):
        return x.pow(n)  # Call Dual's pow() method
    return math_pow(x, n)

# Evaluates a function on a dual number and returns the dual part of result
# Corrosponds to derivative - ie preforms automatic differentiation
def auto_diff(f, x):
    return f(Dual(x, 1)).dual
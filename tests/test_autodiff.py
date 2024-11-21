import pytest
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow
from dual_autodiff import auto_diff

##  This file tests the actual auto_diff function
def test_auto_diff():
    # Tests simple polynomial function f(x) = x^2 + x
    # Expected Result: f(x) = 6, f'(x) = 5, at x=2
    f = lambda x: x**2 + x
    value, derivative = auto_diff(f, 2)
    assert value == 6.0
    assert derivative == 5.0 

    # Tests trigonometric function f(x) = sin(x)
    # Expected Result: f(x) = 0, f'(x) = 1, at x=0
    f = lambda x: x.sin()
    value, derivative = auto_diff(f, 0)
    assert value == 0.0
    assert derivative == 1.0  

    # Tests logarithmic function f(x) = log(x)
    # Expected Result: f(x) = log(2), f'(x) = 0.5, at x=2
    f = lambda x: x.log()
    value, derivative = auto_diff(f, 2)
    assert value == math.log(2)
    assert derivative == 0.5  

    # Tests exponential function f(x) = exp(x)
    # Expected Result: f(x) = exp(1), f'(x) = exp(1), at x=1
    f = lambda x: x.exp()
    value, derivative = auto_diff(f, 1)
    assert value == pytest.approx(math.e)
    assert derivative == pytest.approx(math.e)

    # Tests a more complex function f(x) = x^3 + 3x^2 + 5x + 7
    # Expected Result: f(x) = 16, f'(x) = 14, at x=1
    f = lambda x: x**3 + 3 * x**2 + 5 * x + 7
    value, derivative = auto_diff(f, 1)
    assert value == 16.0
    assert derivative == 14.0  

def test_auto_diff_errors():
    # Tests invalid input for function
    with pytest.raises(TypeError):
        auto_diff("non callable function", 2)

    # Tests invalid input for x
    f = lambda x: x**2 + x
    with pytest.raises(TypeError):
        auto_diff(f, "string")

    # Tests derivative is zero function, f(x) = 7
    # Error which results in non dual output
    # Expected Result: f(x) = 7, f'(x) = 0, at x=3
    f = lambda x: 7
    value, derivative = auto_diff(f, 3)
    assert value == 7.0
    assert derivative == 0.0  # f'(x) = 0
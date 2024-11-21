import pytest
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow
from dual_autodiff import auto_diff

##  This tests the actuall auto_diff function
def test_auto_diff():
    # Test a simple polynomial function f(x) = x^2 + x
    # Expected Result: f'(x) = 2x + 1, at x=2: 2*2 + 1 = 5
    f = lambda x: x**2 + x
    result = auto_diff(f, 2)
    assert result == 5.0 

    # Test a trigonometric function f(x) = sin(x)
    # Expected Result: f'(x) = cos(x), at x=0: cos(0) = 1
    f = lambda x: x.sin()
    result = auto_diff(f, 0)
    assert result == 1.0  

    # Test a logarithmic function f(x) = log(x)
    # Expected Result: f'(x) = 1/x, at x=2: 1/2 = 0.5
    f = lambda x: x.log()
    result = auto_diff(f, 2)
    assert result == 0.5  

    # Test an exponential function f(x) = exp(x)
    # Expected Result: f'(x) = exp(x), at x=1: exp(1)
    f = lambda x: x.exp()
    result = auto_diff(f, 1)
    assert result == pytest.approx(math.e)

    # Test a more complex function f(x) = x^3 + 3x^2 + 5x + 7
    # Expected Result: f'(x) = 3x^2 + 6x + 5, at x=1: 3(1)^2 + 6(1) + 5 = 14
    f = lambda x: x**3 + 3 * x**2 + 5 * x + 7
    result = auto_diff(f, 1)
    assert result == 14.0  

def test_auto_diff_errors():
    # Test invalid input for function
    with pytest.raises(TypeError):
        auto_diff("non callable function", 2)

    # Test invalid input for x
    f = lambda x: x**2 + x
    with pytest.raises(TypeError):
        auto_diff(f, "string")

    # Test a function where the derivative is zero, f(x) = 7
    # Expected Result:
    f = lambda x: 7
    result = auto_diff(f, 3)
    assert result == 0.0  # f'(x) = 0

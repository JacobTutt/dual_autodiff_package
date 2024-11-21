import pytest
import numpy as np
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow
from dual_autodiff import auto_diff

# This file preforms a similiar role to the test_autodiff.py file but with numpy arrays
# Tests the auto_diff function with numpy inputs

def test_auto_diff_numpy():
    # Test simple polynomial function f(x) = x^2 + x
    # Expected values: [2.0, 6.0, 12.0]
    # Expected derivatives: [3.0, 5.0, 7.0]
    f = lambda x: x**2 + x
    x = np.array([1, 2, 3])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [2.0, 6.0, 12.0])
    assert np.allclose(derivative, [3.0, 5.0, 7.0])

    # Test trigonometric function f(x) = sin(x)
    # Expected values: [0.0, 1.0, 0.0]
    # Expected derivatives: [1.0, 0.0, -1.0]
    f = lambda x: x.sin()
    x = np.array([0, np.pi/2, np.pi])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [0.0, 1.0, 0.0])
    assert np.allclose(derivative, [1.0, 0.0, -1.0])

    # Test logarithmic function f(x) = log(x)
    # Expected values: [0.0, math.log(2), math.log(3)]
    # Expected derivatives: [1.0, 0.5, 1/3]
    f = lambda x: x.log()
    x = np.array([1, 2, 3])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [0.0, math.log(2), math.log(3)])
    assert np.allclose(derivative, [1.0, 0.5, 1/3])

    # Test exponential function f(x) = exp(x)
    # Expected values: [1.0, math.exp(1), math.exp(2)]
    # Expected derivatives: [1.0, math.exp(1), math.exp(2)]
    f = lambda x: x.exp()
    x = np.array([0, 1, 2])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [1.0, math.exp(1), math.exp(2)])
    assert np.allclose(derivative, [1.0, math.exp(1), math.exp(2)])

    # Test a more complex function f(x) = x^3 + 3x^2 + 5x + 7
    # Expected values: [16.0, 41.0, 82.0]
    # Expected derivatives: [14.0, 29.0, 50.0]
    f = lambda x: x**3 + 3 * x**2 + 5 * x + 7
    x = np.array([1, 2, 3])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [16.0, 37.0, 76.0])
    assert np.allclose(derivative, [14.0, 29.0, 50.0])

    # Test a more complex function mixing exp, log, and arcsin
    f = lambda x: exp(x**2) + log(x) + arcsin(x / 2)
    x = np.array([0.5, 1, 1.5])
    value, derivative = auto_diff(f, x)
    # Expected values: 0.8436, 3.2419, 10.7413] (rounding errors)
    # Expected derivatives: [3.8004, 7.0139, 29.8858] (rounding errors)

    assert np.allclose(value, [0.8436, 3.2419, 10.7413], rtol = 1e-3)
    assert np.allclose(derivative, [3.8004, 7.0139, 29.8858], rtol = 1e-3)

def test_auto_diff_numpy_errors():
    # Test invalid input for function
    with pytest.raises(TypeError):
        auto_diff("non callable function", np.array([1, 2, 3]))

    # Test invalid input for x
    f = lambda x: x**2 + x
    with pytest.raises(TypeError):
        auto_diff(f, "string")

    # Test derivative is zero function, f(x) = 7
    # Expected values: [7.0, 7.0, 7.0]
    # Expected derivatives: [0.0, 0.0, 0.0]
    f = lambda x: 7
    x = np.array([1, 2, 3])
    value, derivative = auto_diff(f, x)
    assert np.allclose(value, [7.0, 7.0, 7.0])
    assert np.allclose(derivative, [0.0, 0.0, 0.0])

if __name__ == "__main__":
    pytest.main()
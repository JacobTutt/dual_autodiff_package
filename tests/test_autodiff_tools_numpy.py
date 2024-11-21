import pytest
import numpy as np
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow

# This test suite focuses on the package's interaction with numpy arrays of Dual items
# But performs similiar tests to those in the test_autodiff_tools.py file

# Test trigonometric functions with numpy arrays
def test_trig_functions_array():
    x = np.array([Dual(2, 1), Dual(4, 5)])

    # Tests sin function with numpy arrays
    sin_result = sin(x)
    assert math.isclose(sin_result[0].real, math.sin(2))
    assert math.isclose(sin_result[0].dual, math.cos(2))
    assert math.isclose(sin_result[1].real, math.sin(4))
    assert math.isclose(sin_result[1].dual, 5 * math.cos(4))

    # Tests cos function with numpy arrays
    cos_result = cos(x)
    assert math.isclose(cos_result[0].real, math.cos(2))
    assert math.isclose(cos_result[0].dual, -math.sin(2))
    assert math.isclose(cos_result[1].real, math.cos(4))
    assert math.isclose(cos_result[1].dual, -5 * math.sin(4))

    # Tests tan function with numpy arrays
    tan_result = tan(x)
    assert math.isclose(tan_result[0].real, math.tan(2))
    assert math.isclose(tan_result[0].dual, 1 / (math.cos(2) ** 2))
    assert math.isclose(tan_result[1].real, math.tan(4))
    assert math.isclose(tan_result[1].dual, 5 / (math.cos(4) ** 2))

# Tests hyperbolic functions with numpy arrays
def test_hyperbolic_functions_array():
    x = np.array([Dual(2, 1), Dual(4, 5)])

    # Tests sinh function with numpy array  
    sinh_result = sinh(x)
    assert math.isclose(sinh_result[0].real, math.sinh(2))
    assert math.isclose(sinh_result[0].dual, math.cosh(2))
    assert math.isclose(sinh_result[1].real, math.sinh(4))
    assert math.isclose(sinh_result[1].dual, 5 * math.cosh(4))

    # Tests cosh function with numpy array
    cosh_result = cosh(x)
    assert math.isclose(cosh_result[0].real, math.cosh(2))
    assert math.isclose(cosh_result[0].dual, math.sinh(2))
    assert math.isclose(cosh_result[1].real, math.cosh(4))
    assert math.isclose(cosh_result[1].dual, 5 * math.sinh(4))

    # Tests tanh function with numpy array
    tanh_result = tanh(x)
    assert math.isclose(tanh_result[0].real, math.tanh(2))
    assert math.isclose(tanh_result[0].dual, 1 - math.tanh(2) ** 2)
    assert math.isclose(tanh_result[1].real, math.tanh(4))
    assert math.isclose(tanh_result[1].dual, 5 * (1 - math.tanh(4) ** 2))

# Tests inverse trigonometric functions with numpy arrays
def test_inverse_trig_functions_array():
    x = np.array([Dual(0.5, 1), Dual(0.3, 1)])

    # Tests arcsin function with numpy array
    arcsin_result = arcsin(x)
    assert math.isclose(arcsin_result[0].real, math.asin(0.5))
    assert math.isclose(arcsin_result[0].dual, 1 / math.sqrt(1 - 0.5 ** 2))
    assert math.isclose(arcsin_result[1].real, math.asin(0.3))
    assert math.isclose(arcsin_result[1].dual, 1 / math.sqrt(1 - 0.3 ** 2))

    # Tests arccos function with numpy array
    arccos_result = arccos(x)
    assert math.isclose(arccos_result[0].real, math.acos(0.5))
    assert math.isclose(arccos_result[0].dual, -1 / math.sqrt(1 - 0.5 ** 2))
    assert math.isclose(arccos_result[1].real, math.acos(0.3))
    assert math.isclose(arccos_result[1].dual, -1 / math.sqrt(1 - 0.3 ** 2))

    # Tests arctan function with numpy array
    arctan_result = arctan(x)
    assert math.isclose(arctan_result[0].real, math.atan(0.5))
    assert math.isclose(arctan_result[0].dual, 1 / (1 + 0.5 ** 2))
    assert math.isclose(arctan_result[1].real, math.atan(0.3))
    assert math.isclose(arctan_result[1].dual, 1 / (1 + 0.3 ** 2))

# Tests exponential, logarithmic, and power functions with numpy arrays
def test_exp_log_pow_array():
    x = np.array([Dual(2, 1), Dual(4, 5)])

    # Test exponential function with numpy array
    exp_result = exp(x)
    assert math.isclose(exp_result[0].real, math.exp(2))
    assert math.isclose(exp_result[0].dual, math.exp(2))
    assert math.isclose(exp_result[1].real, math.exp(4))
    assert math.isclose(exp_result[1].dual, 5 * math.exp(4))

    # Tests logarithmic function with numpy array
    log_result = log(x)
    assert math.isclose(log_result[0].real, math.log(2))
    assert math.isclose(log_result[0].dual, 1 / 2)
    assert math.isclose(log_result[1].real, math.log(4))
    assert math.isclose(log_result[1].dual, 5 / 4)

    # Tests power function with numpy array
    pow_result = pow(x, 3)
    assert math.isclose(pow_result[0].real, 8)
    assert math.isclose(pow_result[0].dual, 12)
    assert math.isclose(pow_result[1].real, 64)
    assert math.isclose(pow_result[1].dual, 240)

# Tests square root with numpy arrays
def test_sqrt_array():
    x = np.array([Dual(4, 1), Dual(9, 5)])
    sqrt_result = sqrt(x)
    assert math.isclose(sqrt_result[0].real, 2)
    assert math.isclose(sqrt_result[0].dual, 0.25)
    assert math.isclose(sqrt_result[1].real, 3)
    assert math.isclose(sqrt_result[1].dual, 5 / 6)

# Tests invalid inputs with numpy arrays
def test_invalid_inputs_array():
    x = np.array([Dual(-1, 1), Dual(-2, 1)])

    # Tries invalid input with arcsin function
    with pytest.raises(ValueError):
        log(x)

    # Tries invalid exponent type with power function
    with pytest.raises(ValueError):
        sqrt(x)

    # Tries invalid exponent type with power function
    with pytest.raises(TypeError):
        pow(x, "string")


if __name__ == "__main__":
    pytest.main()
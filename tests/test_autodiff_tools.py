import pytest
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow

# Tests trigonometric functions
def test_trig_functions():
    x = Dual(2, 1)

    # Tests sin function for dual numbers
    assert math.isclose(sin(x).real, math.sin(2))
    assert math.isclose(sin(x).dual, math.cos(2))

    # Tests sin function for scalar
    assert math.isclose(sin(2), math.sin(2))

    # Tests cos function for dual numbers
    assert math.isclose(cos(x).real, math.cos(2))
    assert math.isclose(cos(x).dual, -math.sin(2))

    # Tests cos function for scalar
    assert math.isclose(cos(2), math.cos(2))

    # Tests tan function for dual numbers
    assert math.isclose(tan(x).real, math.tan(2))
    assert math.isclose(tan(x).dual, 1 / (math.cos(2) ** 2))

    # Tests tan function for scalar
    assert math.isclose(tan(2), math.tan(2))

    # Tries to find tan where it is undefined
    with pytest.raises(ValueError):
        tan(Dual(math.pi / 2, 1))


# Test hyperbolic functions
def test_hyperbolic_functions():
    x = Dual(1, 1)

    # Test sinh function for dual numbers
    assert math.isclose(sinh(x).real, math.sinh(1))
    assert math.isclose(sinh(x).dual, math.cosh(1))

    # Test sinh function for scalar
    assert math.isclose(sinh(1), math.sinh(1))

    # Test cosh function for dual numbers
    assert math.isclose(cosh(x).real, math.cosh(1))
    assert math.isclose(cosh(x).dual, math.sinh(1))

    # Test cosh function for scalar
    assert math.isclose(cosh(1), math.cosh(1))

    # Test tanh function for dual numbers
    assert math.isclose(tanh(x).real, math.tanh(1))
    assert math.isclose(tanh(x).dual, 1 - math.tanh(1) ** 2)

    # Test tanh function for scalar
    assert math.isclose(tanh(1), math.tanh(1))


# Test inverse trigonometric functions
def test_inverse_trig_functions():
    x = Dual(0.5, 1)

    # Test arcsin function for dual numbers
    assert math.isclose(arcsin(x).real, math.asin(0.5))
    assert math.isclose(arcsin(x).dual, 1 / math.sqrt(1 - 0.5 ** 2))

    # Test arcsin function for scalar
    assert math.isclose(arcsin(0.5), math.asin(0.5))

    # Test acos function for dual numbers
    assert math.isclose(arccos(x).real, math.acos(0.5))
    assert math.isclose(arccos(x).dual, -1 / math.sqrt(1 - 0.5 ** 2))

    # Test acos function for scalar
    assert math.isclose(arccos(0.5), math.acos(0.5))

    # Test atan function for dual numbers
    assert math.isclose(arctan(x).real, math.atan(0.5))
    assert math.isclose(arctan(x).dual, 1 / (1 + 0.5 ** 2))

    # Test atan function for scalar
    assert math.isclose(arctan(0.5), math.atan(0.5))

    # Tries to find arcsin and arccos where it is undefined
    with pytest.raises(ValueError):
        arcsin(Dual(2, 1))


# Test exponential and logarithmic functions
def test_exp_log_functions():
    x = Dual(1, 1)

    # Test exp function for dual numbers
    assert math.isclose(exp(x).real, math.exp(1))
    assert math.isclose(exp(x).dual, math.exp(1))

    # Test exp function for scalar
    assert math.isclose(exp(1), math.exp(1))

    # Test log function for dual numbers
    assert math.isclose(log(x).real, math.log(1))
    assert math.isclose(log(x).dual, 1 / 1)

    # Test log function for scalar
    assert math.isclose(log(1), math.log(1))

    # Tries to find log where it is undefined
    with pytest.raises(ValueError):
        log(Dual(-1, 1))


# Test square root function
def test_sqrt_function():
    x = Dual(4, 1)

    # Test sqrt function for dual numbers
    assert math.isclose(sqrt(x).real, math.sqrt(4))
    assert math.isclose(sqrt(x).dual, 0.5 / math.sqrt(4))

    # Test sqrt function for scalar
    assert math.isclose(sqrt(4), math.sqrt(4))

    # Tries to find sqrt where it is undefined
    with pytest.raises(ValueError):
        sqrt(Dual(-1, 1))


# Test power function
def test_pow_function():
    x = Dual(2, 1)

    # Test pow function for dual numbers
    assert math.isclose(pow(x, 3).real, 2 ** 3)
    assert math.isclose(pow(x, 3).dual, 3 * 2 ** 2)

    # Test pow function for scalar
    assert math.isclose(pow(2, 3), 2 ** 3)

    # Invalid power
    with pytest.raises(TypeError):
        pow(x, "string")
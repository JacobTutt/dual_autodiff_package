import pytest
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow


# This performs similiar tests to those in the test_dual.py file
# But instead of calling the methods directly, it calls the functions from the autodiff_tools.py file which are integrated with the math module's functions
# This test focuses on the the packages interaction with individual Dual items (not numpy)

# Test trigonometric functions
def test_trig_functions():
    x = Dual(2, 1)

    # Test sin function - dual number
    assert math.isclose(sin(x).real, math.sin(2))
    assert math.isclose(sin(x).dual, math.cos(2))

    # Test cos function - dual number
    assert math.isclose(cos(x).real, math.cos(2))
    assert math.isclose(cos(x).dual, -math.sin(2))

    # Test tan function - dual number
    assert math.isclose(tan(x).real, math.tan(2))
    assert math.isclose(tan(x).dual, 1 / (math.cos(2) ** 2))

    # Test tan function with undefined value
    with pytest.raises(ValueError):
        tan(Dual(math.pi / 2, 1))


# Test invalid initialization of trignometry operations
def test_invalid_trig_operations():
    x = Dual(-1, 1)

    # Logarithm of negative real part
    with pytest.raises(ValueError):
        log(x)

    # Square root of negative real part
    with pytest.raises(ValueError):
        sqrt(x)


# Test hyperbolic functions
def test_hyperbolic_functions():
    x = Dual(1, 1)

    # Test sinh function
    assert math.isclose(sinh(x).real, math.sinh(1))
    assert math.isclose(sinh(x).dual, math.cosh(1))

    # Test cosh function
    assert math.isclose(cosh(x).real, math.cosh(1))
    assert math.isclose(cosh(x).dual, math.sinh(1))

    # Test tanh function
    assert math.isclose(tanh(x).real, math.tanh(1))
    assert math.isclose(tanh(x).dual, 1 - math.tanh(1) ** 2)


# Test inverse trigonometric functions
def test_inverse_trig_functions():
    x = Dual(0.5, 1)

    # Test arcsin function
    assert math.isclose(arcsin(x).real, math.asin(0.5))
    assert math.isclose(arcsin(x).dual, 1 / math.sqrt(1 - 0.5 ** 2))

    # Test acos function
    assert math.isclose(arccos(x).real, math.acos(0.5))
    assert math.isclose(arccos(x).dual, -1 / math.sqrt(1 - 0.5 ** 2))

    # Test atan function
    assert math.isclose(arctan(x).real, math.atan(0.5))
    assert math.isclose(arctan(x).dual, 1 / (1 + 0.5 ** 2))

    # Test domain errors
    with pytest.raises(ValueError):
        arcsin(Dual(2, 1))
    with pytest.raises(ValueError):
        arccos(Dual(-2, 1))


# Test exponential, logarithmic, and power functions
def test_exp_log_pow():
    x = Dual(2, 1)

    # Test exponential function
    assert math.isclose(exp(x).real, math.exp(2))
    assert math.isclose(exp(x).dual, math.exp(2))

    # Test logarithmic function
    assert math.isclose(log(x).real, math.log(2))
    assert math.isclose(log(x).dual, 1 / 2)

    # Test power function
    assert (pow(x,3)) == Dual(8.0, 12.0)
    assert (pow(x,0.5)) == Dual(math.sqrt(2), 0.5 / math.sqrt(2))

    # Test logarithm and power errors
    with pytest.raises(ValueError):
        log(Dual(-1, 1))
    with pytest.raises(TypeError):
        pow(x,"string")


# Test square root
def test_sqrt():
    x = Dual(4, 1)
    assert (sqrt(x)) == Dual(2.0, 0.25)
    with pytest.raises(ValueError):
        sqrt(Dual(-1, 1))

if __name__ == "__main__":
    pytest.main()
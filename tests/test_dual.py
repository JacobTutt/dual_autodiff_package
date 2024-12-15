import pytest
import math
from dual_autodiff import Dual

# Tests for initialization and validation (__init__ and _validate_input)
def test_initialization():
    # Test valid inputs
    x = Dual(2, 3)
    assert x.real == 2.0
    assert x.dual == 3.0

    # Test invalid inputs as accounted for in _validate_input
    with pytest.raises(TypeError):
        Dual("string", 2)
    with pytest.raises(ValueError):
        Dual(float("nan"), 2)
    with pytest.raises(ValueError):
        Dual(1, float("inf"))
    with pytest.raises(ValueError):
        Dual(1, float("-inf"))
    # Test against dictionary input called Dual
    with pytest.raises(TypeError):
        Dual(1, {"dual": 2})


# Test for __repr__
def test_repr():
    x = Dual(2, 3)
    assert repr(x) == "Dual(real=2.0, dual=3.0)"


# Test addition operator 
def test_addition():
    x1 = Dual(2, 3)
    x2 = Dual(1, 1)

    # Dual + Dual
    result = x1 + x2
    assert result.real == 3.0
    assert result.dual == 4.0

    # Dual + scalar
    result = x1 + 5
    assert result.real == 7.0
    assert result.dual == 3.0

    # Scalar + Dual
    result = 5 + x1
    assert result.real == 7.0
    assert result.dual == 3.0

    # Invalid addition
    with pytest.raises(TypeError):
        x1 + "string"


# Test subtraction operator
def test_subtraction():
    x1 = Dual(5, 3)
    x2 = Dual(2, 1)

    # Dual - Dual
    result = x1 - x2
    assert result.real == 3.0
    assert result.dual == 2.0

    # Dual - scalar
    result = x1 - 2
    assert result.real == 3.0
    assert result.dual == 3.0

    # Scalar - Dual
    result = 10 - x1
    assert result.real == 5.0
    assert result.dual == -3.0

    # Invalid subtraction
    with pytest.raises(TypeError):
        x1 - "string"


# Test multiplication operator
def test_multiplication():
    x1 = Dual(2, 3)
    x2 = Dual(4, 5)

    # Dual * Dual
    result = x1 * x2
    assert result.real == 8.0
    assert result.dual == 22.0

    # Dual * scalar
    result = x1 * 3
    assert result.real == 6.0
    assert result.dual == 9.0

    # Scalar * Dual
    result = 3 * x1
    assert result.real == 6.0
    assert result.dual == 9.0

    # Invalid multiplication
    with pytest.raises(TypeError):
        x1 * "string"


# Test division operator
def test_division():
    x1 = Dual(6, 4)
    x2 = Dual(2, 1)

    # Dual / Dual
    result = x1 / x2
    assert math.isclose(result.real, 3.0)
    assert math.isclose(result.dual, 0.5)

    # Dual / scalar
    result = x1 / 2
    assert math.isclose(result.real, 3.0)
    assert math.isclose(result.dual, 2.0)

    # Scalar / Dual
    result = 12 / x1
    assert math.isclose(result.real, 2.0)
    assert math.isclose(result.dual, -1.333333333)

    # Test for division by zero account for in dual.py
    with pytest.raises(ZeroDivisionError):
        x1 / 0
    with pytest.raises(ZeroDivisionError):
        Dual(0, 1) / Dual(0, 1)

    # Test for Invalid Type 
    with pytest.raises(TypeError):
        x1 / [1, 2]


def test_power_operator():
    # Dual number raised to an integer power
    x = Dual(2, 1)
    result = x**2
    assert result.real == 4.0
    assert result.dual == 4.0

    # Dual number raised to a float power
    result = x**1.5
    assert pytest.approx(result.real) == 2.8284271247461903  # sqrt(8)
    assert pytest.approx(result.dual) == 2.121320343559643  # (1.5 * sqrt(4) * 1)

    # Edge case: Power of 0
    result = x**0
    assert result.real == 1.0
    assert result.dual == 0.0

    # Edge case: Power of 1
    result = x**1
    assert result.real == 2.0
    assert result.dual == 1.0

    # Dual number raised to a negative power
    result = x**-1
    assert pytest.approx(result.real) == 0.5
    assert pytest.approx(result.dual) == -0.25

    # Invalid power type
    with pytest.raises(TypeError):
        x**"string"

# Test overwritten inequalities
def test_inqualities():
    x1 = Dual(2, 3)
    x2 = Dual(2, 3)
    x3 = Dual(2, 4)
    x4 = Dual(3, 1)

    # Equality and Non-equality
    assert x1 == x2
    assert x1 != x3

    # Less than
    assert x1 < x4
    assert not (x1 < x2)

    # Less than or equal
    assert x1 <= x2
    assert x1 <= x4

    # Greater than
    assert x4 > x1
    assert not (x1 > x2)

    # Greater than or equal
    assert x1 >= x2
    assert x4 >= x1

    # Test for invalid comparison objects 
    with pytest.raises(TypeError):
        x1 < "string"
    with pytest.raises(TypeError):
        x1 <= None


# Test trionmetric functions
def test_trig_functions():
    x = Dual(2, 1)

    # Test sin function
    assert math.isclose(x.sin().real, math.sin(2))
    assert math.isclose(x.sin().dual, math.cos(2))

    # Test cos function
    assert math.isclose(x.cos().real, math.cos(2))
    assert math.isclose(x.cos().dual, -math.sin(2))

    # Test tan function
    assert math.isclose(x.tan().real, math.tan(2))
    assert math.isclose(x.tan().dual, 1 / (math.cos(2) ** 2))

    # Test tan function with undefined value
    with pytest.raises(ValueError):
        Dual(math.pi / 2, 1).tan()


# Test invalid initialisation of operations
def test_invalid_operations():
    x = Dual(-1, 1)

    # Logarithm of negative real part
    with pytest.raises(ValueError):
        x.log()

    # Square root of negative real part
    with pytest.raises(ValueError):
        x.sqrt()


# Test hyperbolic functions
def test_hyperbolic_functions():
    x = Dual(1, 1)

    # Test sinh function
    assert math.isclose(x.sinh().real, math.sinh(1))
    assert math.isclose(x.sinh().dual, math.cosh(1))

    # Test cosh function
    assert math.isclose(x.cosh().real, math.cosh(1))
    assert math.isclose(x.cosh().dual, math.sinh(1))

    # Test tanh function
    assert math.isclose(x.tanh().real, math.tanh(1))
    assert math.isclose(x.tanh().dual, 1 - math.tanh(1) ** 2)

# Test inverse trigonometric functions
def test_inverse_trig_functions():
    x = Dual(0.5, 1)

    # Test arcsin function
    assert math.isclose(x.arcsin().real, math.asin(0.5))
    assert math.isclose(x.arcsin().dual, 1 / math.sqrt(1 - 0.5 ** 2))

    # Test acos function
    assert math.isclose(x.arccos().real, math.acos(0.5))
    assert math.isclose(x.arccos().dual, -1 / math.sqrt(1 - 0.5 ** 2))

    # Test atan function
    assert math.isclose(x.arctan().real, math.atan(0.5))
    assert math.isclose(x.arctan().dual, 1 / (1 + 0.5 ** 2))

    # Test domain errors
    with pytest.raises(ValueError):
        Dual(2, 1).arcsin()
    with pytest.raises(ValueError):
        Dual(-2, 1).arccos()


# Test exponential, logarithmic, and power functions
def test_exp_log_pow():
    x = Dual(2, 1)

    # Test exponential function
    assert math.isclose((x.exp()).real, math.exp(2))
    assert math.isclose((x.exp()).dual, math.exp(2))

    # Test logarithmic function
    assert math.isclose(x.log().real, math.log(2))
    assert math.isclose(x.log().dual, 1 / 2)

    # Test power function
    assert (x.pow(3)) == Dual(8.0, 12.0)
    assert (x.pow(0.5)) == Dual(math.sqrt(2), 0.5 / math.sqrt(2))

    # Test logarithm and power errors
    with pytest.raises(ValueError):
        Dual(-1, 1).log()
    with pytest.raises(TypeError):
        x.pow("string")


# Test square root
def test_sqrt():
    x = Dual(4, 1)
    assert (x.sqrt()) == Dual(2.0, 0.25)
    with pytest.raises(ValueError):
        Dual(-1, 1).sqrt()

if __name__ == "__main__":
    pytest.main()
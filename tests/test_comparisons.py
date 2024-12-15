import pytest
import numpy as np
from dual_autodiff import Dual

# Test comparisons of Dual numbers
def test_dual_comparisons():
    x = Dual(2, 1)
    y = Dual(2, 1)
    z = Dual(3, 1)

    # Test equality
    assert x == y
    assert not x == z

    # Test inequality
    assert x != z
    assert not x != y

    # Test less than
    assert x < z
    assert not x < y

    # Test less than or equal
    assert x <= y
    assert x <= z

    # Test greater than
    assert z > x
    assert not x > y

    # Test greater than or equal
    assert x >= y
    assert z >= x

# Test invalid type entries 
def test_rsub_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        "string" - x  

# Division by zero
def test_rtruediv_zero_real():
    x = Dual(0, 1)
    with pytest.raises(ZeroDivisionError):
        1 / x  

# Dividing string
def test_rtruediv_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        "string" / x 

# Test invalid comparison woith sting
def test_eq_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        x == "string" 
def test_ne_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        x != "string" 
def test_gt_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        x > "string"  
def test_ge_invalid_type():
    x = Dual(2, 1)
    with pytest.raises(TypeError):
        x >= "string" 

# Test comparisons of arrays of Dual numbers
def test_dual_array_comparisons():
    x = np.array([Dual(2, 1), Dual(3, 1), Dual(1, 1)])
    y = np.array([Dual(2, 1), Dual(3, 1), Dual(1, 1)])
    z = np.array([Dual(3, 1), Dual(4, 1), Dual(0, 1)])

def test_dual_array_comparisons():
    x = np.array([Dual(2, 1), Dual(3, 1), Dual(1, 1)])
    y = np.array([Dual(2, 1), Dual(3, 1), Dual(1, 1)])
    z = np.array([Dual(3, 1), Dual(4, 1), Dual(0, 1)])

    # Test equality
    assert np.array_equal(x == y, [True, True, True])
    assert np.array_equal(x == z, [False, False, False])

    # Test inequality
    assert np.array_equal(x != z, [True, True, True])
    assert np.array_equal(x != y, [False, False, False])

    # Test less than
    assert np.array_equal(x < z, [True, True, False])
    assert np.array_equal(x < y, [False, False, False])

    # Test less than or equal
    assert np.array_equal(x <= y, [True, True, True])
    assert np.array_equal(x <= z, [True, True, False])

    # Test greater than
    assert np.array_equal(z > x, [True, True, False])
    assert np.array_equal(x > y, [False, False, False])

    # Test greater than or equal
    assert np.array_equal(x >= y, [True, True, True])
    assert np.array_equal(z >= x, [True, True, False])

if __name__ == "__main__":
    pytest.main()
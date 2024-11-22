import pytest
import numpy as np
import math
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, arcsin, arccos, arctan, pow
from dual_autodiff import multi_auto_diff

# This test suite focuses on the package's interaction with numpy arrays of Dual items
# and tests the multi_auto_diff function

# Test a more complex already testing in test_autodiff_numpy.py to check function works as it should in reduntant case
def test_multi_auto_diff_complex_function():
    funcs = [lambda x: exp(x**2) + log(x) + arcsin(x / 2)]

    x = np.array([0.5, 1, 1.5])
    results = multi_auto_diff(funcs, x)
    
    expected_real = np.array([0.8436, 3.2419, 10.7413])
    expected_dual = np.array([3.8004, 7.0139, 29.8858])
    
    assert np.allclose(results[0][0], expected_real, rtol = 1e-3)
    assert np.allclose(results[0][1], expected_dual, rtol = 1e-3)

# Test multiple functions with numpy arrays
def test_multi_auto_diff_numpy():
    funcs = [
        lambda x: x**2 + x,
        lambda x: x.sin(),
        lambda x: x.log(),
        lambda x: x.exp(),
        lambda x: x**3 + 3 * x**2 + 5 * x + 7]
    
    x = np.array([1, 2, 3])
    results = multi_auto_diff(funcs, x)
    
    # Expected results
    expected_real = [
        np.array([2.0, 6.0, 12.0]),
        np.array([math.sin(1), math.sin(2), math.sin(3)]),
        np.array([math.log(1), math.log(2), math.log(3)]),
        np.array([math.exp(1), math.exp(2), math.exp(3)]),
        np.array([16.0, 37.0, 76.0])]
    
    expected_dual = [
        np.array([3.0, 5.0, 7.0]),
        np.array([math.cos(1), math.cos(2), math.cos(3)]),
        np.array([1.0, 0.5, 1/3]),
        np.array([math.exp(1), math.exp(2), math.exp(3)]),
        np.array([14.0, 29.0, 50.0])]
    
    for result, expected_r, expected_d in zip(results, expected_real, expected_dual):
        assert np.allclose(result[0], expected_r)
        assert np.allclose(result[1], expected_d)

# Test invalid inputs with numpy arrays
def test_multi_auto_diff_invalid_inputs():
    funcs = [
        lambda x: x**2 + x,
        "non callable function"
    ]
    x = np.array([1, 2, 3])

    # Test invalid function
    with pytest.raises(TypeError):
        multi_auto_diff(funcs, x)

    # Test invalid input for x
    funcs = [lambda x: x**2 + x]
    with pytest.raises(TypeError):
        multi_auto_diff(funcs, "string")

if __name__ == "__main__":
    pytest.main()
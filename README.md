# Dual Numbers in Dual Autodiff Package
[**`dual_autodiff`**](https://dual-autodiff-package.readthedocs.io/en/latest/) is a Python library that enables the use of dual numbers with the aim of preforming automatic differentiation.

### Recent Updates - Version v1.1.0

The recent release of `dual_autodiff` provides extensive integration with `numpy` arrays, allowing the user to seamlessly work with arrays of dual numbers.

- **Enhanced Mathematical Functions**: All mathematical functions such as `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, and more are now fully compatible with `numpy` arrays of dual numbers .
- **Advanced Auto-differentiation Functions**: Extended `auto_diff` to be fully compatible with `numpy` arrays, enabling automatic differentiation over arrays of dual numbers.
- **Multi-Function Differentiation**: Introduced the `multi_auto_diff` function, allowing users to evaluate multiple functions and their derivatives at once, and even over a range (array) of points.
- **Advanced Error Handling and Testing**: Improved error handling mechanisms and added comprehensive test coverage to ensure robustness and reliability.

## Motivation

**Dual numbers** provide a mathematically robust way to compute derivatives automatically and exactly during function evaluation. It typically has lower computational overhead than numerical approaches and eliminates the associated approximation errors. 

ie. consider $f(x) = x^2$:
- Input: Dual number - $x + \epsilon$
- Output: 

$$f(x) = (x + \epsilon)^2 = x^2 + 2x\epsilon$$

- $x^2$: Output's real value - the functions value
- $2x$: Output's dual value - the functions derivative

### Applications:
- **Optimisation**: In algorithms like gradient descent.
- **Machine Learning**: Enabling backpropagation and training of neural networks.
- **Physics and Engineering**: For solving differential equations.

## Features

All features now fully compatible with **numpy arrays of Dual numbers**

- **Dual Numbers**: A class to store dual numbers
- **Arithmetic Operations** for dual numbers
  - Addition, subtraction: `+`, `-`
  - Multiplication, and division: `*`, `/`
- **Comparison Operations** for dual numbers
  - Equal and not equal: `=`, `!=`
  - Less than (or equal to): `<`, `<=`
  - Less than (or equal to): `>`, `>=`
- **Mathematical Functions**:
  - Trigonometric: `sin`, `cos`, `tan`, and their inverses (`arcsin`, `arccos`, `arctan`).
  - Hyperbolic: `sinh`, `cosh`, `tanh`.
  - Exponential and logarithmic: `exp`, `log`.
  - Powers and roots: `pow`, `sqrt`.
- **Automatic Differentiation**: Compute derivatives automatically using the properties of dual numbers.
  - `auto_diff(func, value)`
- **Multi-Function Differentiation**: Evaluate multiple functions and their derivatives at once.
  - `multi_auto_diff(funcs, value)`
- **dual_autodiff**: A comprehensive Jupyter Notebook showcasing the features and usage of the package.

## Installation

1. Clone the repository:
    - Clone the repository from the remote repository to your local machine.
    ```bash
    git clone https://github.com/JacobTutt/dual_autodiff_package.git
    cd dual_autodiff_package
    ```

2. Install the package:

    2.1. Install the full package:
    - For general use
    ```bash
    pip install -e .
    ```
    - For optional dependecies (ie to run testing and example notebooks)
    ```bash
    pip install -e ".[tutorial]" # to be able to run notebook
    pip install -e ".[testing]" # to be able to run test suit
    pip install -e. ".[docs]" # to be able to build docs locally
    pip install -e ".[tutorial,testing,docs]" # to be able to run all (for coursework assesment)
    ```

    2.2. Or install binary wheels:
    eg.
    - For `cp310-manylinux_x86_64` (Python 3.10 on Linux):
    ```bash
    pip install wheelhouse/dual_autodiff-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    ```
      - similarly to include optional dependencies
      ```bash
      pip install "wheelhouse/dual_autodiff-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl[tutorial,testing,docs]"
      ```

## Usage

### Import the Package
##### Normal python version
```python
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, ...
from dual_autodiff import auto_diff
```

##### Cythonized version
- The cythonised version stored in the subpackage `dual_autodiff_x` fully support all functionality and typically runs 10-20% faster.
```python
from dual_autodiff_x import Dual
from dual_autodiff_x import sin, cos, tan, ...
from dual_autodiff_x import auto_diff
```

### Basic Examples

##### Initialising Dual Numbers
```python
x = Dual(2, 1)   # Dual number: 2 + 1ε
y = Dual(3, 2)   # Dual number: 3 + 2ε
```

##### Arithmetic Operations
```python
print(x + y)     # Dual(real=5, dual=3)
print(x / y)     # Dual(real=0.666..., dual=-0.222...)
```

##### Comparison Operations
```python
print(x == y)    # False
print(x != y)    # True
print(x < y)     # True
```

##### Mathematical Functions in Class
```python
print(x.sin())   # Dual(real=0.909..., dual=-0.416...)
print(x.log())   # Dual(real=0.693..., dual=0.5)
```

##### Mathematical Functions using 'Math' operators
```python
print(pow(x, n)) # Dual(real=8.0, dual=12.0)
print(atan(x))   # Dual(real=1.107..., dual=0.2)
```

##### Mathematical Functions in 'numpy' array
```python
x = np.array([Dual(2, 1), Dual(4, 5)])
print(sin(x))    #[Dual(real=0.909..., dual=-0.416...),Dual(real=-0.757..., dual=-3.268...)]
print(exp(x))    ##[Dual(real=7.389..., dual=7.389...), Dual(real=54.598..., dual=272.991...)]
```

##### Automatic Differentiation - auto_diff
```python
func = x**2 + 3*x
print(auto_diff(func, 5)) # 13.0
```

##### Automatic Differentiation with numpy arrays - auto_diff
```python
func = lambda x: x**2 + 3*x
x = np.array([1, 2, 3])
value, derivative = auto_diff(func, x)
print(value)        # [ 4. 10. 18.]
print(derivative)   # [ 5.  7.  9.]
```

##### Multi-Function Differentiation - multi_auto_diff
```python
funcs = [
    lambda x: x**2 + x,
    lambda x: sin(x)
    lambda x: log(x)
]
x = np.array([1, 2, 3])
results = multi_auto_diff(funcs, x)
for value, derivative in results:
    print("Value:", value)
    print("Derivative:", derivative)

# Expected output:
# Value: [ 2.  6. 12.]
# Derivative: [3. 5. 7.]
# Value: [0.841..., 0.909..., 0.141...]
# Derivative: [0.540..., -0.416..., -0.990...]
# Value: [0.   , 0.693..., 1.099...]
# Derivative: [1.   , 0.5   , 0.333...]
```




### More comprehensive Examples

- For more comprehensive examples see:
  -  **dual_autodiff.ipynb** in the tutorials folder
  - the packages documentation

## Testing

To ensure the package has installed correctly you may want to run the tests for the `dual_autodiff` package

1. **Install the package with testing dependencies**:
    ```bash
    pip install -e ".[testing]"
    ```

2. **Run the tests using `pytest`**:
    ```bash
    pytest
    ```

## Tutorial

To interactively explore the features and usage of the `dual_autodiff` package, you can use the provided tutorial in the Jupyter notebook.

1. **Install the package with tutorial dependencies**:
    ```bash
    pip install -e ".[tutorial]"
    ```
2. **Generate Jupyter Kernel for current enviroment**:
    ```bash
    python -m ipykernel install --user --name=env --display-name "Python (dual_autodiff)"
    ```

3. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook tutorials/dual_autodiff.ipynb
    ```
  - or navigate to and run using your chosen IDE

## Documentation

- Please find full documentation at the [Read the Docs Documentation](https://dual-autodiff-package.readthedocs.io/en/latest/)

## Contributing
To contribute to this package:
- 1. Fork the repository.
- 2. Create a new branch for your feature or bugfix.
- 3. Set up pre-commit hooks:
    This helps run automatic testing to ensure consistency with current package features before commiting
    - Install `pre-commit`:
      ```bash
      pip install pre-commit
      ```
    - Set up the hooks:
      ```bash
      pre-commit install
      ```
- 4. Submit a pull request.


## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Support
If you need help or have any questions, please open an issue on the GitHub Issues page.

### Report
Please find a report about the packages development under the `report` directory

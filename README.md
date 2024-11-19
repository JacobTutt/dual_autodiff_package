# Dual Numbers in Dual Autodiff Package
**`dual_autodiff`** is a Python library that enables the use of dual numbers in the aim of preforming automatic differentiation.

## Motivation

**Dual numbers** provide a mathematically robust way to compute derivatives automatically and exactly during function evaluation. It typically has lower computational overhead than numerical approaches and eliminates the associated approximation errors. 

ie. consider $f(x) = x^2$:
- Input: Dual number - $x + \epsilon$
- Output: 

$$f(x) = (x + \epsilon)^2 = x^2 + 2x\epsilon$$

- $x^2$: Outputs real value - the functions value
- $2x$: Outputs dual value - the functions derivative

### Applications:
- **Optimization**: In algorithms like gradient descent.
- **Machine Learning**: Enabling backpropagation and training of neural networks.
- **Physics and Engineering**: For solving differential equations.

## Features

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
- **dual_autodiff**: A comprehensive Jupyter Notebook showcasing the features and usage of the package.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/JacobTutt/dual_autodiff_package.git
    cd dual_autodiff_package
    ```

2. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

3. For the testing, install optional dependencies:
    ```bash
    pip install '.[testing]'
    ```

4. For the tutorial, install optional dependencies:
    ```bash
    pip install '.[tutorial]'
    ```

## Usage

### Import the Package
- Normal python version
```python
from dual_autodiff import Dual
from dual_autodiff import sin, cos, tan, ...
from dual_autodiff import auto_diff
```

- Cythonized version
```python
from dual_autodiff_x import Dual
from dual_autodiff_x import sin, cos, tan, ...
from dual_autodiff_x import auto_diff
```

### Basic Examples
```python
## Initialise two dual numbers x, y
x = Dual(2, 1)   # Dual number: 2 + 1ε
y = Dual(3, 2)   # Dual number: 3 + 2ε

# Arithmetic Operations
print(x + y)     # Dual(real=5, dual=3)
print(x - y)     # Dual(real=-1, dual=-1)
print(x * y)     # Dual(real=6, dual=7)
print(x / y)     # Dual(real=0.666..., dual=-0.222...)

# Mathematical Functions in Class
print(x.sin())   # Dual(real=0.9092..., dual=-0.4161...)
print(x.log())   # Dual(real=0.6931..., dual=0.5)

## Mathematical Functions using Math operators
print(pow(x, n)) # Dual(real=8.0, dual=12.0)
print(atan(x))   # Dual(real=1.1071487177940904, dual=0.2)

## Automatic Differentiation 
func = x**2 + 3*x
print(auto_diff(func, 5)) # 13.0
```

### More comprehensive Examples

- For more comprehensive examples see: **dual_autodiff.ipynb** in the tutorials folder

### Documentation

- Please find full documentation at the [Read the Docs Documentation](https://dual-autodiff-package.readthedocs.io/en/latest/)

## Contributing
To contribute to this package:
- 1. Fork the repository.
- 2. Create a new branch for your feature or bugfix.
- 3. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.
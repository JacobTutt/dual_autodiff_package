Overview
========



What are Dual Numbers
---------------------

**Dual numbers** are those expressed in the form:

.. math::

   x + \epsilon y

Where:

- :math:`x`: the **real part**.
- :math:`y`: the **dual part** (can represent the derivative).
- :math:`\epsilon`: a quantity with the defining property :math:`\epsilon^2 = 0`.

Why Are Dual Numbers Useful?
----------------------------

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual numbers are used within **forward-mode automatic differentiation**, 
a method for computing **exact derivatives of functions**. This method 
has low computational overhead and can simultaneously compute a 
function's value :math:`f(x)` and its derivative :math:`f'(x)`.

For a function :math:`f(x)`, using dual numbers, we can compute both the 
value of the function and its derivative simultaneously. Consider 
:math:`f(x) = x^2`:

- **Input**: Dual number - :math:`x + \epsilon`
- **Output**:

.. math::

   f(x) = (x + \epsilon)^2 = x^2 + 2x\epsilon

- :math:`x^2`: Outputs the real value - the function's value.
- :math:`2x`: Outputs the dual value - the function's derivative.

Applications
~~~~~~~~~~~~

- **Optimization**: In algorithms like gradient descent.
- **Machine Learning**: Enabling backpropagation and training of neural networks.
- **Physics and Engineering**: For solving differential equations.

Package Features
----------------
All features now fully compatible with **numpy arrays of Dual numbers**

- **Dual Numbers**: A class to store dual numbers
- **Arithmetic Operations** for dual numbers
  - Addition, subtraction: ``+``, ``-``
  - Multiplication, and division: ``*``, ``/``
- **Comparison Operations** for dual numbers
  - Equal and not equal: ``=``, ``!=``
  - Less than (or equal to): ``<``, ``<=``
  - Less than (or equal to): ``>``, ``>=``
- **Mathematical Functions**:
  - Trigonometric: ``sin``, ``cos``, ``tan``, and their inverses (``arcsin``, ``arccos``, ``arctan``).
  - Hyperbolic: ``sinh``, ``cosh``, ``tanh``.
  - Exponential and logarithmic: ``exp``, ``log``.
  - Powers and roots: ``pow``, ``sqrt``.
- **Automatic Differentiation**: Compute derivatives automatically using the properties of dual numbers.
  - ``auto_diff(func, value)``
- **Multi-Function Differentiation**: Evaluate multiple functions and their derivatives at once.
  - ``multi_auto_diff(funcs, value)``
- **dual_autodiff**: A comprehensive Jupyter Notebook showcasing the features and usage of the package.

User Guide - Installation
=========================

Method 1: Build Package
-----------------------

1. Clone the repository:

   .. code:: bash

      git clone https://github.com/JacobTutt/dual_autodiff_package.git
      cd dual_autodiff_package

2. Install the package:

   .. code:: bash

      pip install .

Method 2: Use Pre-Built Wheels (Cython)
---------------------------------------

1. Clone the repository:

   .. code:: bash

      git clone https://github.com/JacobTutt/dual_autodiff_package.git
      cd dual_autodiff_package

2. Install the specific wheel for your platform and Python version:

   - For ``cp310-manylinux_x86_64`` (Python 3.10 on Linux):

     .. code:: bash

        pip install wheelhouse/dual_autodiff-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

   - For ``cp311-manylinux_x86_64`` (Python 3.11 on Linux):

     .. code:: bash

        pip install wheelhouse/dual_autodiff-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

Usage
-----

Import the Package
~~~~~~~~~~~~~~~~~~

To use the package, simply import it into your Python scripts:

.. code:: python

   import dual_autodiff 
   import dual_autodiff_x # For Cython converted package when using Method 1


Optional package features
=========================

Testing
-------

The package includes a test suite to ensure the installation is correct and the package functions as expected.

1. Install optional dependencies for the testing:

   .. code:: bash

      pip install '.[testing]'

2. Run the tests:

   .. code:: bash

      pytest

Tutorial Notebook
-----------------

The package includes a tutorial notebook to demonstrate the features and usage of the package, the user is encouraged to interact with this notebook to understand the packages functionality, errors it may raise and how to use it effectively.

1. To access the tutorial notebook, install the optional dependencies:

   .. code:: bash

      pip install '.[tutorial]'

2. Open the Jupyter notebook:

   .. code:: bash

      jupyter notebook notebooks/dual_autodiff_tutorial.ipynb

Package Examples
================

Further indepth examples including errors and exceptions can be found in:
- **dual_autodiff.ipynb** in the tutorials directory.
- API Reference section in the documentation.

Initialise Dual Numbers
-----------------------

.. code:: python

   from dual_autodiff import Dual
   ## Initialize two dual numbers x, y
   x = Dual(2, 1)  # Dual number: 2 + 1ε
   y = Dual(3, 2)  # Dual number: 3 + 2ε

Basic Arithmetic Operations
----------------------------

.. code:: python

   ## Addition
   z = x + y  # Dual number: 5 + 3ε
   ## Subtraction
   z = x - y  # Dual number: -1 - 1ε
   ## Multiplication
   z = x * y  # Dual number: 6 + 7ε
   ## Division
   z = x / y  # Dual number: 2/3 + 1/3ε
   ## Powers
   z = x ** 3  # Dual number: 8 + 12ε

Basic Arithmetic Operations with numpy arrays
----------------------------------------------

.. code:: python

   import numpy as np
   from dual_autodiff import Dual

   ## Initialise numpy arrays of Dual numbers
   x = np.array([Dual(2, 1), Dual(4, 5)])
   y = np.array([Dual(3, 2), Dual(1, 1)])

   ## Addition
   z = x + y # [Dual(real=5, dual=3), Dual(real=5, dual=6)]
   ## Subtraction
   z = x - y # [Dual(real=-1, dual=-1), Dual(real=3, dual=4)]
   ## Multiplication
   z = x * y # [Dual(real=6, dual=7), Dual(real=4, dual=9)]
   ## Division
   z = x / y # [Dual(real=0.667..., dual=-0.111...), Dual(real=4.0, dual=1.0)]
   ## Powers
   z = x ** 2
   print("x ** 2:", z) # [Dual(real=4, dual=4), Dual(real=16, dual=40)]

Basic Comparison Operations
----------------------------

.. code:: python

   ## Equal
   x == y  # False
   ## Not Equal
   x != y  # True
   ## Less Than
   x < y  # True
   ## Less Than or Equal To
   x <= y  # True
   ## Greater Than
   x > y  # False
   ## Greater Than or Equal To
   x >= y  # False

Mathematical Functions
----------------------

- The package supports mathematical operators from the dual class or by importing math functions.

Directly from the dual class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   ## Trigonometric Functions
   x.sin()  # Dual number: sin(2) + 1cos(2)ε
   ## Inverse Trigonometric Functions
   x.arccos()  # Dual number: arccos(2) - 1/sqrt(1-2^2)ε
   ## Hyperbolic Functions
   x.tanh()  # Dual number: tanh(2) + 1sech^2(2)ε
   ## Exponential and Logarithmic Functions
   x.exp()  # Dual number: exp(2) + 1exp(2)ε
   ## Powers and Roots
   x.pow(3)  # Dual number: 2^3 + 3*2^2ε


Using math functions
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from dual_autodiff import Dual, cos, arctan, log, sqrt

   ## Trigonometric Functions
   cos(x) # Dual number: cos(2) - 1sin(2)ε
   ## Inverse Trigonometric Functions
   arctan(x) # Dual number: arctan(2) + 1/(1+2^2)ε
   ## Hyperbolic Functions
   sinh(x) # Dual number: sinh(2) + 1cosh(2)ε
   ## Exponential and Logarithmic Functions
   log(x) # Dual number: log(2) + 1/2ε
   ## Powers and Roots
   sqrt(x) # Dual number: sqrt(2) + 1/(2sqrt(2))ε

Mathematical Functions in 'numpy' array
---------------------------------------

.. code:: python

   import numpy as np
   from dual_autodiff import Dual, tan, arcsin, cosh, exp, sqrt

   ## Initialise numpy arrays of Dual numbers
   x = np.array([Dual(2, 1), Dual(4, 5)]) 
   ## Trigonometric Functions
   tan(x) # [Dual(real=-2.185..., dual=5.774...), Dual(real=1.158..., dual=34.730...)]
   ## Inverse Trigonometric Functions
   arctan(x) # Dual(real=1.107..., dual=0.2  ), Dual(real=1.326..., dual=0.294...)]
   ## Hyperbolic Functions
   cosh(x) # [Dual(real=3.762..., dual=3.627...), Dual(real=27.308..., dual=136.450...)]
   ## Exponential and Logarithmic Functions
   exp(x) # [Dual(real=7.389..., dual=7.389...),Dual(real=54.598..., dual=272.991...)]
   ## Powers and Roots
   sqrt(x) # [Dual(real=1.414..., dual=0.353...), Dual(real=2.0, dual=1.25)]

Automatic Differentiation
-------------------------

- Some examples demonstrating how to use the `auto_diff` function from the `dual_autodiff` package to compute derivatives of various functions.
- The dual_autodiff package currently supports a wide range of mathematical functions, but some may be in development. 

Derivative of :math:`f(x) = x^2`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mathematical Concept
^^^^^^^^^^^^^^^^^^^^

The derivative of the function :math:`f(x) = x^2` is calculated as follows:

.. math::

   f'(x) = \frac{d}{dx} (x^2) = 2x

So, the derivative at :math:`x = 2` is:

.. math::

   f'(2) = 2 \cdot 2 = 4

Code Example
^^^^^^^^^^^^

.. code:: python

   from dual_autodiff import auto_diff

   def func(x):
       return x ** 2

   ## Compute the derivative of the function at x = 2
   derivative_at_2 = auto_diff(func, 2)  # 4

Derivative of :math:`g(x) = \log(\sin(x)) + \exp(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mathematical Concept
^^^^^^^^^^^^^^^^^^^^

The derivative of the function :math:`g(x) = \log(\sin(x)) + \exp(x)`  is calculated as follows:

.. math::

   g'(x) = \frac{d}{dx} (\log(\sin(x)) + \exp(x)) = \frac{\cos(x)}{\sin(x)} + \exp(x) 

So, the derivative at :math:`x = 2` is:

.. math::

   g'(2) = \frac{\cos(2)}{\sin(2)} + \exp(2) 

Code Example
^^^^^^^^^^^^

.. code:: python

   from dual_autodiff import auto_diff

   ## Define a function
   def func_2(x):
       return log(sin(x)) + exp(x)

   ## Compute the derivative of the function at x = 2
   derivative_at_2 = auto_diff(func_2, 2)  # cos(2)/sin(2) + exp(2)

Automatic Differentiation with numpy arrays - auto_diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from dual_autodiff import auto_diff

   def func_2(x):
       return log(sin(x)) + exp(x)

   x = np.array([1, 2, 3])

   ## Compute the derivative of the function at x = 1, 2, 3
   value, derivative = auto_diff(func_2, x)
   print(value)        # [ 2.545..., 7.293... , 18.127...]
   print(derivative)   # [ 3.360..., 6.931..., 13.070...]

   
Multi-Function Differentiation - multi_auto_diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from dual_autodiff import multi_auto_diff
   import numpy as np

   funcs = [
       lambda x: x**2 + x,
       lambda x: sin(x),
       lambda x: log(x)
   ]

   x = np.array([1, 2, 3])

   ## Compute the derivative of func1, func2, func3 at x = 1, 2, 3
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


Tutorial Notebook
-----------------

For more comprehensive examples, see the **dual_autodiff_tutorial** in the notebooks directory.
- Instructions on how to run the tutorial notebook are provided in the **Usage** section.

Developer Notes:
================

Contributing
------------

To contribute to this package:

1. Fork the repository.

2. Create a new branch for your feature or bugfix.

3. Submit a pull request.

License
-------

This project is licensed under the MIT License. See the LICENSE file for more information.

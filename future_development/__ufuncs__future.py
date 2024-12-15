# This would normally not be included in the main repository, especially if the package, however for the purpose of this project, it is included to show the future development of the package.

def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    Handle NumPy ufuncs (universal functions) for Dual objects.
    Supported unfuncs: 
        - Addition: np.add()
        - Subtraction: np.subtract()
        - Multiplication: np.multiply()
        - Division: np.divide()
        - Sine: np.sin()
        - Cosine: np.cos()
        - Tangent: np.tan()
        - Arcsin: np.arcsin()
        - Arccos: np.arccos()
        - Arctan: np.arctan()
        - Sinh: np.sinh()
        - Cosh: np.cosh()
        - Tanh: np.tanh()
        - Exponential: np.exp()
        - Logarithm: np.log()
        - Powers: np.pow(n)
        - Square Root: np.sqrt()
        - Equal: np.equal()
        - Not Equal: np.not_equal()
        - Less: np.less()
        - Less Equal: np.less_equal()
        - Greater: np.greater()
        - Greater Equal: np.greater_equal()

    Returns:
        Dual or NotImplemented: The result of the ufunc operation, or NotImplemented if the operation is not supported.

    Example:
        >>> x = Dual(2, 1)
        >>> y = Dual(3, 2)
        >>> np.add(x, y)
        Dual(real=5.0, dual=3.0)
        >>> np.multiply(x, y)
        Dual(real=6.0, dual=7.0)
    """

    if method != '__call__':
        return NotImplemented

    # Extract real and dual parts from inputs
    real_parts = []
    dual_parts = []
    for x in inputs:
        if isinstance(x, Dual):
            real_parts.append(x.real)
            dual_parts.append(x.dual)
        else:  # Treat scalars or standard arrays as real numbers with dual=0
            real_parts.append(x)
            dual_parts.append(0)

    
    # Convert lists to arrays for broadcasting
    real_parts = np.array(real_parts)
    dual_parts = np.array(dual_parts)


    # Now define the operations for each option of ufunc 

    # add and sub are element-wise operations and therefore can keep unfunc as operation to support vectorization
    if ufunc in {np.add, np.subtract}:
        real_result = ufunc(*real_parts, **kwargs)
        dual_result = ufunc(*dual_parts, **kwargs)

    # Multiplication
    elif ufunc == np.multiply:
        # Real part is the product of the real parts
        result_real = np.prod(real_parts, axis=0)

        # Calculates the dual parts (ie first order coefficients) of the product by summing the products of the real parts excluding the current index
    
        # non vectorized version below for reference
        # result_dual = 0  # Initialize the result for the dual part

        # # Loop through each dual component
        # for i, dual in enumerate(duals):
        #     # Compute the product of the real parts excluding the current index
        #     product_of_reals = 1  # Start with a multiplicative identity
        #     for j, r in enumerate(reals):
        #         if j != i:  # Exclude the current index
        #             product_of_reals *= r
        
        # # Add the contribution of the current dual to the total dual result
        # result_dual += dual * product_of_reals

        # Vectorized version exploiting numpy's efficeint broadcasting
        result_dual = sum( (dual * np.prod([r for j, r in enumerate(real_parts) if j != i],axis=0)) for i, dual in enumerate(dual_parts))

    # Division
    # As np.divide does not accept variafic arguments, only two inputs should be supported and vectorisation is not required
    elif ufunc == np.divide:  # Division
        # Ensures only two inputs for division
        if len(inputs) != 2:
            raise ValueError("np.divide only supports two inputs (numerator and denominator).")

        # Real parts divided by respectively
        result_real = real_parts[0] / real_parts[1]

        # Dual parts calculated as before
        result_dual = (dual_parts[0] * real_parts[1] - real_parts[0] * dual_parts[1]) / (real_parts[1] ** 2)


    # Trigonometric functions
    # Done part wise and using numpy's vectorization
    # Use same logic as _dual_function

    #sin
    if ufunc == np.sin:
        real_result = np.sin(real_parts)
        dual_result = np.cos(real_parts) * dual_parts
    
    #cos
    elif ufunc == np.cos:
        real_result = np.cos(real_parts)
        dual_result = -np.sin(real_parts) * dual_parts

    #tan
    elif ufunc == np.tan:  
        real_result = np.tan(real_parts)
        dual_result = (1 / np.cos(real_parts) ** 2) * dual_parts

    # Inverse trigonometric functions
    # Done part wise and using numpy's vectorization
    # Use same logic as _dual_function

    #arcsin
    elif ufunc == np.arcsin:
        real_result = np.arcsin(real_parts)
        dual_result = dual_parts / np.sqrt(1 - real_parts ** 2)

    #arccos
    elif ufunc == np.arccos: 
        real_result = np.arccos(real_parts)
        dual_result = -dual_parts / np.sqrt(1 - real_parts ** 2)

    #arctan
    elif ufunc == np.arctan:
        real_result = np.arctan(real_parts)
        dual_result = dual_parts / (1 + real_parts ** 2)

    # Hyperbolic functions
    # Done part wise and using numpy's vectorization
    # Use same logic as _dual_function

    # sinh
    elif ufunc == np.sinh:
        real_result = np.sinh(real_parts)
        dual_result = np.cosh(real_parts) * dual_parts

    # cosh
    elif ufunc == np.cosh:
        real_result = np.cosh(real_parts)
        dual_result = np.sinh(real_parts) * dual_parts

    # tanh
    elif ufunc == np.tanh:
        real_result = np.tanh(real_parts)
        dual_result = (1 - np.tanh(real_parts) ** 2) * dual_parts

    # Exponential, Logarithmic and Square root functions
    # Done part wise and using numpy's vectorization
    # Use same logic as _dual_function

    # exp
    elif ufunc == np.exp:  
        real_result = np.exp(real_parts)
        dual_result = np.exp(real_parts) * dual_parts

    # log
    elif ufunc == np.log:  
        real_result = np.log(real_parts)
        dual_result = dual_parts / real_parts

    # Square Root
    elif ufunc == np.sqrt:
        real_result = np.sqrt(real_parts)
        dual_result = 0.5 * dual_parts / np.sqrt(real_parts)

    # Power
    elif ufunc == np.power:  # Power
        if len(inputs) != 2:
            raise ValueError("np.power only supports two inputs (base and exponent).")
        if not isinstance(inputs[1], (int, float)):
            raise ValueError(f"Exponent must be an int/float for the use of np.power with Dual class, got type {type(inputs[1]).__name__}.")
        
        real_result = np.power(real_parts[0], real_parts[1])
        dual_result = real_parts[1] * np.power(real_parts[0], real_parts[1] - 1) * dual_parts[0]

    # Comparison operators
    # Apart from  (not) equal to are done on real parts only
    elif ufunc in {np.less, np.less_equal, np.greater, np.greater_equal}:
        return ufunc(*real_parts, **kwargs)
    
    # not equal to and equal to are done on both real and dual parts
    elif ufunc in {np.equal, np.not_equal}:
        return ufunc(*real_parts, **kwargs) and ufunc(*dual_parts, **kwargs)

    else:
        raise NotImplementedError(f"Ufunc {ufunc} not implemented for Dual numbers")

    # Return final result of all operations
    return np.array([Dual(r, d) for r, d in zip(real_result, dual_result)])

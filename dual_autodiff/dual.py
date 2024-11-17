import math

class Dual:
    """
    A class to represent dual numbers, enabling basic operations
    and automatic differentiation.

    """
    # Initialising the 'Dual number' class
    def __init__(self, real: float, dual: float):
        """
        Initialize a dual number with its real and dual parts.

        Parameters:
            real (float): The real part of the dual number.
            dual (float): The dual part of the dual number.
        """

        # Preforming input validation using seperate class function to allow reuse
        self.real = self._validate_input(real, "Real part")
        self.dual = self._validate_input(dual, "Dual part")

    @staticmethod
    def _validate_input(part_value: float, part: str):
        """
        Validate the input for the real or dual part.

        Parameters:
            part_value (float): The input value to validate.
            part (str): The part of the dual number being validated.

        Returns:
            float: The validated value.

        Raises:
            TypeError: If 'real part' or 'dual part' is not a number.
            ValueError: If 'real part' or 'dual part' is NaN or infinite.
        """
        # Check value is of form of an integer or float
        if not isinstance(part_value, (int, float)):
            raise TypeError(f"'{part}' must be a number (int or float), got {type(part_value).__name__}.")
        
        # Check value is not a 'NaN' - 'floating point number' for undefined values
        # If so throw ValueError
        if math.isnan(part_value):  
            raise ValueError(f"'{part}' cannot be NaN.")
        
        # Check value is not for infite (+/-)
        # If so throw ValueError
        if part_value in (float("inf"), float("-inf")):  
            raise ValueError(f"'{part}' cannot be infinite.")
        
        # Ensure if correct input it is cast to float
        return float(part_value) 
    


    # Use of '__repr__' to set a dual numbers output format 'Dual(real = ..., dual = ...)'
    def __repr__(self):
        """
        Output the dual number in a readable format.

        Returns:
            str: A string describing the dual number.
        """
        return f"Dual(real={self.real}, dual={self.dual})"
    


    # Overwrites addition operator for use on dual numbers 
    # Written to support addition of dual numbers together and dual numbers with scalar
    def __add__(self, other):
        """
        Define addition for Dual numbers and scalars.

        This method supports addition in both standard and reverse cases:
            - Dual + Dual
            - Dual + scalar
            - scalar + Dual (via __radd__)

        Parameters:
            other (Dual, int, or float): The value to add to the current Dual number.

        Returns:
            Dual: A new Dual number representing the result of the addition.

        Raises:
            TypeError: If 'other' is not a Dual, int, or float.

        Examples:
            1. Dual + Dual:
            >>> x = Dual(2, 1)
            >>> y = Dual(3, 2)
            >>> x + y
            Dual(real=5, dual=3)

            2. Dual + scalar:
            >>> x = Dual(2, 1)
            >>> x + 3
            Dual(real=5, dual=1)

            3. Scalar + Dual:
            >>> x = Dual(2, 1)
            >>> 3 + x
            Dual(real=5, dual=1)

            4. Invalid input:
            >>> x = Dual(2, 1)
            >>> x + "string"
            Traceback (most recent call last):
                ...
            TypeError: Addition is only supported with Dual or scalar values.
        """

        # If both operands are dual numbers add real and dual parts respectively
        # (a+bε) + (c+dε) = (a+c) + (b+d)ε
        if isinstance(other, Dual):
            real_part = self.real + other.real
            dual_part = self.dual + other.dual
            return Dual(real_part, dual_part)
        
        # If dual + int/float
        # (a+bε) + c = (a+c) + bε
        elif isinstance(other, (int, float)):
            return Dual(self.real + other, self.dual)
        
        # If not addition of dual and scalar - throw 'Type Error'
        else:
            raise TypeError("Addition failed as only supported with Dual or scalar values.")
        
    # This overwrites the reverse addition opertor to support int/float + dual
    # a + (c+dε) = (a+c) + dε
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise TypeError("Addition failed as only supported with Dual or scalar values.")
    


    # Overwrites subtraction operator for use on dual numbers 
    # Written to support subtraction of dual numbers together and dual numbers with scalar
    def __sub__(self, other):
        """
        Define subtraction for Dual numbers and scalars.

        This method supports subtraction in both standard and reverse cases:
            - Dual - Dual
            - Dual - scalar
            - scalar - Dual (via __rsub__)

        Parameters:
            other (Dual, int, or float): The value to subtract from the current Dual number.

        Returns:
            Dual: A new Dual number representing the result of the subtraction.

        Raises:
            TypeError: If 'other' is not a Dual, int, or float.

        Examples:
            1. Dual - Dual:
            >>> x = Dual(5, 3)
            >>> y = Dual(2, 1)
            >>> x - y
            Dual(real=3, dual=2)

            2. Dual - scalar:
            >>> x = Dual(5, 3)
            >>> x - 2
            Dual(real=3, dual=3)

            3. Scalar - Dual:
            >>> x = Dual(5, 3)
            >>> 10 - x
            Dual(real=5, dual=-3)

            4. Invalid input:
            >>> x = Dual(5, 3)
            >>> x - "string"
            Traceback (most recent call last):
                ...
            TypeError: Subtraction is only supported with Dual or scalar values.
        """

        # If both operands are dual numbers subtract real and dual parts respectively
        # (a+bε) - (c+dε) = (a-c) + (b-d)ε
        if isinstance(other, Dual):
            real_part = self.real - other.real
            dual_part = self.dual - other.dual
            return Dual(real_part, dual_part)
        
        # If dual - int/float
        # (a+bε) - c = (a-c) + bε
        elif isinstance(other, (int, float)):
            return Dual(self.real - other, self.dual)
        
        # If not subtraction of dual and scalar - throw 'Type Error'
        else:
            raise TypeError("Subtraction failed as only supported with Dual or scalar values.")
        
    # This overwrites the reverse subtraction opertor to support int/float - dual
    # a - (c+dε) = (a-c) - dε
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.dual)  
        else:
            raise TypeError("Subtraction failed as only supported with Dual or scalar values.")



    # Overwrites multiplication operator for use on dual numbers 
    # Written to support multiplication of dual numbers together and dual numbers with scalar
    def __mul__(self, other):
        """
        Define multiplication for Dual numbers and scalars.

        This method supports multiplication in both standard and reverse cases:
            - Dual * Dual
            - Dual * scalar
            - scalar * Dual (via __rmul__)

        Parameters:
            other (Dual, int, or float): The value to multiply with the current Dual number.

        Returns:
            Dual: A new Dual number representing the result of the multiplication.

        Raises:
            TypeError: If 'other' is not a Dual, int, or float.

        Examples:
            1. Dual * Dual:
            >>> x = Dual(5, 3)
            >>> y = Dual(2, 1)
            >>> x * y
            Dual(real=10, dual=11)

            2. Dual * scalar:
            >>> x = Dual(5, 3)
            >>> x * 2
            Dual(real=10, dual=6)

            3. Scalar * Dual:
            >>> x = Dual(5, 3)
            >>> 2 * x
            Dual(real=10, dual=6)

            4. Invalid input:
            >>> x = Dual(5, 3)
            >>> x * "string"
            Traceback (most recent call last):
                ...
            TypeError: Multiplication is only supported with Dual or scalar values.
        """

        # If both operands are dual numbers:
        # (a + bε) * (c + dε) = ac + (ad + bc)ε
        if isinstance(other, Dual):
            real_part = self.real * other.real
            dual_part = self.real * other.dual + self.dual * other.real
            return Dual(real_part, dual_part)
        
        # If dual * int/float
        # (a + bε) * c  = a*c + (b*c)ε
        elif isinstance(other, (int, float)):
            return Dual(self.real * other, self.dual * other)
        
        # If not multiplication of dual and scalar - throw 'TypeError'
        else:
            raise TypeError("Multiplication failed as only supported with Dual or scalar values.")
        
    # This overwrites the reverse multiplication opertor to support int/float * dual
    # a - (c+dε) = (a-c) - dε
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError("Multiplication failed as only supported with Dual or scalar values.")
    
    

    def __truediv__(self, other):
        """
        Define division for Dual numbers and scalars.

        This method supports division in both standard and reverse cases:
            - Dual / Dual
            - Dual / scalar
            - scalar / Dual (via __rtruediv__)

        Parameters:
            other (Dual, int, or float): The value to divide by.

        Returns:
            Dual: A new Dual number representing the result of the division.

        Raises:
            TypeError: If 'other' is not a Dual, int, or float.
            ZeroDivisionError: If dividing by zero (in the real part for Dual, or scalar zero).

        Examples:
            1. Dual / Dual:
            >>> x = Dual(6, 4)
            >>> y = Dual(2, 1)
            >>> x / y
            Dual(real=3.0, dual=0.5)

            2. Dual / scalar:
            >>> x = Dual(6, 4)
            >>> x / 2
            Dual(real=3.0, dual=2.0)

            3. Scalar / Dual:
            >>> x = Dual(6, 4)
            >>> 12 / x
            Dual(real=2.0, dual=-1.3333...)

            4. Invalid input:
            >>> x = Dual(6, 4)
            >>> y = Dual(0, 1)
            >>> x / y
            Traceback (most recent call last):
                ...
            ZeroDivisionError: Cannot divide by a Dual number with no real part.
        """
        # If both operands are Dual numbers:
        # (a + bε) / (c + dε) = (a/c) + ((b*c - a*d) / c^2)ε
        if isinstance(other, Dual):
            # If division by 0 (undefined) - throws ZeroDivisionError
            if other.real == 0:
                raise ZeroDivisionError("Cannot divide by a Dual number with no real part.")
            real_part = self.real / other.real
            dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return Dual(real_part, dual_part)
        
        # If dividing a Dual number by a scalar (int or float):
        # (a + bε) / c = (a/c) + (b/c)ε
        elif isinstance(other, (int, float)):
            # If division by 0 (undefined) - throws ZeroDivisionError
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return Dual(self.real / other, self.dual / other)
        
        # If not division of dual and scalar - throw 'TypeError'
        else:
            raise TypeError("Division is only supported with Dual or scalar values.")

    # This overwrites the reverse multiplication opertor to support int/float / dual
    # a / (c+dε) = (a / c) - (a * d / c^2)ε
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if self.real == 0:
                raise ZeroDivisionError("Cannot divide by a Dual number with zero as its real part.")
            real_part = other / self.real
            dual_part = (-self.dual * other) / (self.real ** 2)
            return Dual(real_part, dual_part)
        else:
            raise TypeError("Division is only supported with Dual or scalar values.")
        


    # Defines a generic function implementing the preperty of dual numbers
    # f(a+bε) = f(a) = f'(a)*bε
    # Used internally by specific functions which input function and its derivative
    def _dual_function(self, func, func_deriv):
        """
        Internal method to apply a function to a dual number.

        Supported Functions: 
            - Sine: .sin()
            - Cosine: .cos()
            - Tangent: .tan()
            - Arcsin: .arcsin()
            - Arccos: .arccos()
            - Arctan: .arctan()
            - Sinh: .sinh()
            - Cosh: .cosh()
            - Tanh: .tanh()
            - Exponential: .exp()
            - Logarithm: .log()
            - Powers: .pow(n)
            
        Parameters:
            func (callable): The mathematical function applied to dual number.
            func_deriv (callable): The derivative of the function.

        Returns:
            Dual: The out the function on a dual number
        """
        real_part = func(self.real)
        dual_part = func_deriv(self.real) * self.dual
        return Dual(real_part, dual_part)
    


    # Defines Sin() operator using _dual_function
    # Derivative: Cos()
    def sin(self):
        """
        Compute the sine of the Dual number.

        Returns:
            Dual: Sine of original dual number

        Examples:
            >>> x = Dual(2, 1)
            >>> x.sin()
            Dual(real=0.9092..., dual=-0.4161...)
        """
        return self._dual_function(math.sin, math.cos)

    # Defines Cos() operator using _dual_function
    # Derivative: -Sin()
    def cos(self):
        """
        Compute the cosine of the Dual number.

        Returns:
            Dual: Cosine of original dual number

        Examples:
            >>> x = Dual(2, 1)
            >>> x.cos()
            Dual(real=-0.4161..., dual=-0.9093...)
        """
        return self._dual_function(math.cos, lambda x: -math.sin(x))
    
    # Defines tan() operator using _dual_function - log(x)
    # Derivative: 1/cos^2(x)
    def tan(self):
        """
        Compute the tangent of the Dual number.

        Returns:
            Dual: Tangent of original dual number

        Raises:
            ValueError: If the tangent function is undefined as cosine of real part equals 0

        Examples:
            >>> x = Dual(2, 1)
            >>> x.tan()
            Dual(real=-2.1850..., dual=5.7744...)

            >>> Dual(math.pi / 2, 1).tan()
            Traceback (most recent call last):
                ...
            ValueError: Tangent undefined when cosine of real part equals 0.
        """

        # Checks for regions in which tan is undefined - eg. pi/2
        # Same regions in which cos(x) = 0 - uses tolerance of 1e-9 around value
        # If case: throws 'ValueError'
        if math.isclose(math.cos(self.real), 0, abs_tol=1e-9):
            raise ValueError("Tangent undefined when cosine of real part equals 0.")
        return self._dual_function(math.tan, lambda x: 1 / (math.cos(x) ** 2))

    # Defines exp() operator using _dual_function - e^x
    # Derivative: exp()
    def exp(self):
        """
        Compute the exponential of the Dual number.

        Returns:
            Dual: Expontential of original dual number

        Examples:
            >>> x = Dual(2, 1)
            >>> x.exp()
            Dual(real=7.3891..., dual=7.3891...)
        """
        return self._dual_function(math.exp, math.exp)

    # Defines log() operator using _dual_function - log(x)
    # Derivative: 1/x
    def log(self):
        """
        Compute the natural logarithm of the Dual number.

        Returns:
            Dual: Natural logirithim of original dual number

        Raises:
            ValueError: If the real part of dual number is non-positive

        Examples:
            >>> x = Dual(2, 1)
            >>> x.log()
            Dual(real=0.6931..., dual=0.5)

            >>> x = Dual(0, 1)
            >>> x.log()
            Traceback (most recent call last):
                ...
            ValueError: Logarithm is undefined for dual numbers with non-positive real parts.
        """

        # Checks for Logirithm of non positive number for which undefined
        # If so throws 'Value Error'
        if self.real <= 0:
            raise ValueError("Logarithm is undefined for dual numbers with non-positive real parts.")
        return self._dual_function(math.log, lambda x: 1 / x)
    
    # Defines arcsin() operator using _dual_function
    # Derivative: 1/sqrt(1-x^2)
    def arcsin(self):
        """
        Compute the arcsine (inverse sine) of the Dual number.

        Returns:
            Dual: Arcsine of the original Dual number.

        Raises:
            ValueError: If the real part is outside the range [-1, 1].

        Examples:
            >>> x = Dual(0.5, 1)
            >>> x.asin()
            Dual(real=0.5236..., dual=1.1547...)

            >>> x = Dual(1.5, 1)
            >>> x.asin()
            Traceback (most recent call last):
                ...
            ValueError: Arcsine is only defined for real parts in the range [-1, 1].
        """

        # Checks for defined inputs where arcsin is defined: [-1,1]
        # If so throws 'Value Error'
        if not -1 <= self.real <= 1:
            raise ValueError("Arcsine is only defined for real parts in the range [-1, 1].")
        return self._dual_function(math.asin, lambda x: 1 / math.sqrt(1 - x**2))
    
    # Defines arccos() operator using _dual_function
    # Derivative: -1/sqrt(1-x^2)
    def arccos(self):
        """
        Compute the arccosine (inverse cosine) of the Dual number.

        Returns:
            Dual: Arccosine of the original Dual number.

        Raises:
            ValueError: If the real part is outside the range [-1, 1].

        Examples:
            >>> x = Dual(0.5, 1)
            >>> x.acos()
            Dual(real=1.0472..., dual=-1.1547...)

            >>> x = Dual(-1.5, 1)
            >>> x.acos()
            Traceback (most recent call last):
                ...
            ValueError: Arccosine is only defined for real parts in the range [-1, 1].
        """
        # Checks for defined inputs where arcccos is defined: [-1,1]
        # If so throws 'Value Error'
        if not -1 <= self.real <= 1:
            raise ValueError("Arccosine is only defined for real parts in the range [-1, 1].")
        return self._dual_function(math.acos, lambda x: -1 / math.sqrt(1 - x**2))

    # Defines arctan() operator using _dual_function
    # Derivative: 1/sqrt(1+x^2)
    def arctan(self):
        """
        Compute the arctangent (inverse tangent) of the Dual number.

        Returns:
            Dual: Arctangent of the original Dual number.

        Examples:
            >>> x = Dual(1, 1)
            >>> x.atan()
            Dual(real=0.7854..., dual=0.5)
        """
        return self._dual_function(math.atan, lambda x: 1 / (1 + x**2))
    

    # Defines hyperbolic operator: sinh() using _dual_function
    # Derivative: hyperbolic operator: cosh()
    def sinh(self):
        """
        Compute the hyperbolic sine of the Dual number.

        Returns:
            Dual: Hyperbolic sine of the original Dual number.

        Examples:
            >>> x = Dual(1, 1)
            >>> x.sinh()
            Dual(real=1.1752..., dual=1.5431...)
        """
        return self._dual_function(math.sinh, math.cosh)

    # Defines hyperbolic operator: cosh() using _dual_function
    # Derivative: hyperbolic operator: sinh()
    def cosh(self):
        """
        Compute the hyperbolic cosine of the Dual number.

        Returns:
            Dual: Hyperbolic cosine of the original Dual number.

        Examples:
            >>> x = Dual(1, 1)
            >>> x.cosh()
            Dual(real=1.5431..., dual=1.1752...)
        """
        return self._dual_function(math.cosh, math.sinh)
    
    # Defines hyperbolic operator: tanh() using _dual_function
    # Derivative: hyperbolic operator: 1-tanh^2()
    def tanh(self):
        """
        Compute the hyperbolic tangent of the Dual number.

        Returns:
            Dual: Hyperbolic tangent of the original Dual number.

        Examples:
            >>> x = Dual(1, 1)
            >>> x.tanh()
            Dual(real=0.7616..., dual=0.4200...)
        """
        return self._dual_function(math.tanh, lambda x: 1 - math.tanh(x)**2)
    
    # Defines square root operator - sqrt() using _dual_function
    # Derivative: (1/2) x^(-1/2)
    def sqrt(self):
        """
        Compute the square root of the Dual number.

        Returns:
            Dual: Square root of the original dual number.

        Raises:
            ValueError: If the real part of dual number is negative.

        Examples:
            >>> x = Dual(4, 1)
            >>> x.sqrt()
            Dual(real=2.0, dual=0.25)

            >>> x = Dual(0, 1)
            >>> x.sqrt()
            Traceback (most recent call last):
                ...
            ValueError: Square root is undefined when real parts of dual number are negetive.
        """
        if self.real < 0:
            raise ValueError("Square root is undefined when real parts of dual number are negetive.")
        return self._dual_function(math.sqrt, lambda x: 0.5 / math.sqrt(x))
    
    # Defines 'to the power of' operator - pow(n) using _dual_function
    # Derivative: n * x^(n-1)
    def pow(self, n):
        """
        Compute the Dual number raised to a power n.

        Parameters:
            n (float): The power to which the Dual number is raised.

        Returns:
            Dual: The original dual number raised to the given power.

        Examples:
            >>> x = Dual(2, 1)
            >>> x.pow(3)
            Dual(real=8.0, dual=12.0)
        """
        return self._dual_function(lambda x: x**n, lambda x: n * x**(n-1))



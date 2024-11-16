import math

class Dual:
    """
    A class to represent dual numbers, enabling basic operations
    and automatic differentiation.

    """

    def __init__(self, real: float, dual: float):
        """
        Initialize a dual number with its real and dual parts.

        Parameters:
            real (float): The real part of the dual number.
            dual (float): The dual part of the dual number.
        """
        self.real = self._validate_input(real, "Real part")
        self.dual = self._validate_input(dual, "Dual part")

    def _validate_input(part_value: float, part: str) -> float:
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
        if not isinstance(part_value, (int, float)):
            raise TypeError(f"'{part}' must be a number (int or float), got {type(part_value).__name__}.")
        if math.isnan(part_value):  # Check for NaN
            raise ValueError(f"'{part}' cannot be NaN.")
        if part_value in (float("inf"), float("-inf")):  # Check for infinity
            raise ValueError(f"'{part}' cannot be infinite.")
        return float(part_value)  # Ensure the value is cast to float

    def __repr__(self) -> str:
        """
        Output the dual number in a readable format.

        Returns:
            str: A string describing the dual number.
        """
        return f"Dual(real={self.real}, dual={self.dual})"
    
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
        if isinstance(other, Dual):
            real_part = self.real + other.real
            dual_part = self.dual + other.dual
            return Dual(real_part, dual_part)
        elif isinstance(other, (int, float)):
            return Dual(self.real + other, self.dual)  # Add the scalar to the real part
        else:
            raise TypeError("Addition failed as only supported with Dual or scalar values.")
        
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise TypeError("Addition failed as only supported with Dual or scalar values.")
    
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
        if isinstance(other, Dual):
            real_part = self.real - other.real
            dual_part = self.dual - other.dual
            return Dual(real_part, dual_part)
        elif isinstance(other, (int, float)):
            return Dual(self.real - other, self.dual)
        else:
            raise TypeError("Subtraction failed as only supported with Dual or scalar values.")

    def __rsub__(self, other):
        # Delegate reverse subtraction logic to handle scalar - Dual
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.dual)  # Reverse subtraction logic
        else:
            raise TypeError("Subtraction failed as only supported with Dual or scalar values.")

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
        if isinstance(other, Dual):
            # Multiplication rule for Dual numbers: (a + bε) * (c + dε) = ac + (ad + bc)ε
            real_part = self.real * other.real
            dual_part = self.real * other.dual + self.dual * other.real
            return Dual(real_part, dual_part)
        elif isinstance(other, (int, float)):
            # Scalar multiplication: Scale both real and dual parts
            return Dual(self.real * other, self.dual * other)
        else:
            raise TypeError("Multiplication failed as only supported with Dual or scalar values.")

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
        if isinstance(other, Dual):
            if other.real == 0:
                raise ZeroDivisionError("Cannot divide by a Dual number with no real part.")
            real_part = self.real / other.real
            dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return Dual(real_part, dual_part)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return Dual(self.real / other, self.dual / other)
        else:
            raise TypeError("Division is only supported with Dual or scalar values.")

    def __rtruediv__(self, other):
        # Delegate scalar / Dual division logic to __truediv__
        if isinstance(other, (int, float)):
            if self.real == 0:
                raise ZeroDivisionError("Cannot divide by a Dual number with zero as its real part.")
            real_part = other / self.real
            dual_part = (-self.dual * other) / (self.real ** 2)
            return Dual(real_part, dual_part)
        else:
            raise TypeError("Division is only supported with Dual or scalar values.")
        
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
        if self.real <= 0:
            raise ValueError("Logarithm is undefined for dual numbers with non-positive real parts.")
        return self._dual_function(math.log, lambda x: 1 / x)

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
        if math.isclose(math.cos(self.real), 0, abs_tol=1e-9):
            raise ValueError("Tangent undefined when cosine of real part equals 0.")
        return self._dual_function(math.tan, lambda x: 1 / (math.cos(x) ** 2))
    
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
        if not -1 <= self.real <= 1:
            raise ValueError("Arcsine is only defined for real parts in the range [-1, 1].")
        return self._dual_function(math.asin, lambda x: 1 / math.sqrt(1 - x**2))

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
        if not -1 <= self.real <= 1:
            raise ValueError("Arccosine is only defined for real parts in the range [-1, 1].")
        return self._dual_function(math.acos, lambda x: -1 / math.sqrt(1 - x**2))

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

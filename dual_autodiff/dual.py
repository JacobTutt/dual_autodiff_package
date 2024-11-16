import math

class Dual:
    """
    A class to represent dual numbers, enabling basic operations
    and forward-mode automatic differentiation.

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
        
    
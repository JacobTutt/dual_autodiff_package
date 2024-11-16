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

        Raises:
            TypeError: If 'real' or 'dual' is not a number.
            ValueError: If 'real' or 'dual' is NaN or infinite.
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
            TypeError: If 'part_value' is not a number.
            ValueError: If 'part_value' is NaN or infinite.
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

        This method defines addition between:
        1. Two Dual numbers:
        Example: Dual(2, 1) + Dual(3, 2) → Dual(real=5, dual=3)

        2. A Dual number and a scalar (int or float):
        Example: Dual(2, 1) + 3 → Dual(real=5, dual=1)

        2. Invalid input:
        Example: Dual(2, 1) + "string" → Raises TypeError: "Addition is only supported with Dual or scalar values."

        Parameters:
            other (Dual, int, or float): The value to add to the current Dual number.

        Returns:
            Dual: A new Dual number representing the result of the addition.

        Raises:
            TypeError: If 'other' is not a Dual, int, or float.
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
        """
        This method supports adding a scalar (int or float) to a Dual number when the scalar is on the left-hand side of the `+` operator.

        Examples:
        1. Scalar + Dual:
        Example: 3 + Dual(2, 1) → Dual(real=5, dual=1)

        2. Invalid input:
        Example: "string" + Dual(2, 1) → Raises TypeError: "Addition is only supported with Dual or scalar values."

        Parameters:
            other (int or float): The scalar value to add.

        Returns:
            Dual: A new Dual number representing the result of the addition.

        Raises:
            TypeError: If 'other' is not a scalar (int or float).
        """
        if isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise TypeError("Addition failed as only supported with Dual or scalar values.")
        

    def __sub__(self, other):
        """
        Define subtraction for dual numbers.
        (a + bε) - (c + dε) = (a - c) + (b - d)ε
        """
        if not isinstance(other, Dual):
            raise TypeError("Subtraction is only supported between two Dual objects.")
        real_part = self.real - other.real
        dual_part = self.dual - other.dual
        return Dual(real_part, dual_part)
    
    
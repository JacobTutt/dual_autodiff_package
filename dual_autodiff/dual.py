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
    
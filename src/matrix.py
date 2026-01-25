from typing import Any, List, Type, Tuple
import copy

from src.fields import Field


class Matrix:
    """A matrix containing field elements"""
    _matrix: List[List[Any]]
    _field: Field

    def __init__(self, matrix: List[List[Any]], field: Type[Field] | Field):
        self._field = field() if isinstance(field, type) else field

        for row in matrix:
            for idx, element in enumerate(row):
                # convert to field element (e.g. int to Fraction) and verify it is a field element
                element = self._field.instantiate(element)
                row[idx] = element

        self._matrix = matrix

    @classmethod
    def identity(cls, n: int, m: int, field: Field) -> 'Matrix':
        """Create an n x m identity matrix over the given field."""
        result: List[List[Any]] = []
        for i in range(n):
            row: List[Any] = []
            for j in range(m):
                if i == j:
                    row.append(field.one())
                else:
                    row.append(field.zero())
            result.append(row)
        return cls(result, field)
    def validate_index(self, row: int, col: int) -> None:
        """Validate that the given row and column indices are within the matrix bounds."""
        if row < 0 or row >= len(self._matrix):
            raise IndexError(f"Row index out of range. (input: {row})")
        if col < 0 or col >= len(self._matrix[0]):
            raise IndexError(f"Column index out of range. (input: {col})")

    def swap_rows(self, row1: int, row2: int) -> None:
        """Swap two rows of the matrix (ERO 1)"""
        self.validate_index(row1, 0)
        self.validate_index(row2, 0)
        self._matrix[row1], self._matrix[row2] = self._matrix[row2], self._matrix[row1]

    def multiply_row(self, row_indx: int, scalar: Any) -> None:
        try:
            scalar = self._field.instantiate(scalar)
        except ValueError as e:
            raise ValueError(f"Invalid scalar for multiplication: {scalar}") from e
        self.validate_index(row_indx, 0)
        for col in range(self.cols):
            self._matrix[row_indx][col] = self._matrix[row_indx][col] * scalar

    def add_multiple_of_row(self, target_row: int, source_row: int, scalar: Any) -> None:
        try:
            scalar = self._field.instantiate(scalar)
        except ValueError as e:
            raise ValueError(f"Invalid scalar for row addition: {scalar}") from e
        self.validate_index(target_row, 0)
        self.validate_index(source_row, 0)
        for col in range(self.cols):
            product = self._matrix[source_row][col] * scalar
            self._matrix[target_row][col] = self._matrix[target_row][col] + product

    def first_nonzero(self, col: int, first_row: int = 0) -> int:
        """Returns the index of the first nonzero element in a column, or -1 if all are zero."""
        self.validate_index(0, col)
        if first_row < 0 or first_row >= self.rows:
            raise IndexError(f"First row index out of range. (first_row: {first_row})")
        for row in range(first_row, self.rows):
            if self._matrix[row][col] != self._field.zero():
                return row
        return -1

    def get_pivot(self, col: int) -> int:
        """
        Returns the row index of the pivot in the column,
        or -1 if none exists.
        Assumes the matrix is in RREF.
        """
        self.validate_index(0, col)
        found_pivot = -1
        for row in range(self.rows):
            if self._matrix[row][col] != self._field.zero():
                is_pivot = True
                for c in range(0, col):
                    if c != col and self._matrix[row][c] != self._field.zero():
                        is_pivot = False
                        break
                if is_pivot:
                    found_pivot = row
                    break
        return found_pivot

    def mult(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication: self * other"""
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions for multiplication.")
        if not isinstance(other.field , type(self.field)):
            raise ValueError("Matrices must be over the same field for multiplication.")

        result_data: List[List[Any]] = []
        for i in range(self.rows):
            result_row: List[Any] = []
            for j in range(other.cols):
                sum_element = self._field.zero()
                for k in range(self.cols):
                    product = self._matrix[i][k] * other[k, j]
                    sum_element = sum_element + product
                result_row.append(sum_element)
            result_data.append(result_row)

        return Matrix(result_data, self._field)


    @property
    def rows(self) -> int:
        """Number of rows in the matrix."""
        return len(self._matrix)
    @property
    def cols(self) -> int:
        """Number of columns in the matrix."""
        return len(self._matrix[0])

    @property
    def matrix(self) -> List[List[Any]]:
        """Return a copy of the underlying matrix"""
        return copy.deepcopy(self._matrix)

    @property
    def field(self) -> Field:
        return self._field

    def __str__(self) -> str:
        #individual elements need to have str() called to print properly
        result = ""
        for row in self._matrix:
            result += "[" + ", ".join(str(elem) for elem in row) +  "]\n"
        return result.strip()

    def __getitem__(self, key: int | Tuple[int, int]) -> Any:
        """
        Supports:
        m[row] -> returns a copy of the row list
        m[row, col] -> returns the element at (row, col)
        """
        # Row access
        if isinstance(key, int):
            self.validate_index(key, 0)
            return copy.deepcopy(self._matrix[key])
        else:
        # Element access
            row, col = key
            self.validate_index(row, col)
            return self._matrix[row][col]

    def __setitem__(self, key: Any, value: Any) -> None:
        #don't want to allow direct access to underlying matrix to add validation

        """
        Supports:
        m[row] = [...row values...]
        m[row, col] = value
        """
        # Row assignment
        if isinstance(key, int):
            index = key
            self.validate_index(index, 0)
            if not isinstance(value, list):
                raise TypeError("Row assignment requires a list.")
            if len(value) != self.cols: #type: ignore
                raise ValueError(
                    f"Row length must be {self.cols}. (input length: {len(value)})" #type: ignore
                    )
            new_row: List[Any] = []
            for element in value: #type: ignore
                try:
                    new_row.append(self._field.instantiate(element))
                except ValueError as e:
                    raise ValueError(f"Invalid field element: {element}") from e
            self._matrix[index] = new_row
            return

        # Element assignment
        if isinstance(key, tuple) and len(key) == 2: #type: ignore
            row, col = key #type: ignore
            self.validate_index(row, col) #type: ignore
            try:
                element = self._field.instantiate(value)
            except ValueError as e:
                raise ValueError(f"Invalid field element: {value}") from e
            self._matrix[row][col] = element
            return


        raise IndexError("Invalid key type. Use m[row] or m[row, col].")

from typing import Any, List, Type, Tuple
import copy

from src.main.fields import Field


#TODO: make this a container class (need to change field classes first)
class Matrix:
    """A matrix modeled off of SymPy's DomainMatrix, but micropython compatible"""
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
        """
        Swap two rows of the matrix (ERO type 1).
        
        :param row1: First row index
        :type row1: int
        :param row2: Second row index
        :type row2: int
        """
        self.validate_index(row1, 0)
        self.validate_index(row2, 0)
        self._matrix[row1], self._matrix[row2] = self._matrix[row2], self._matrix[row1]

    def scale_row(self, row_indx: int, scalar: Any) -> None:
        """
        Scale a row of the matrix (ERO type 2).
        
        :param row_indx: Row index to scale
        :type row_indx: int
        :param scalar: Scalar to multiply the row by
        :type scalar: Any
        """
        try:
            scalar = self._field.instantiate(scalar)
        except ValueError as e:
            raise ValueError(f"Invalid scalar for multiplication: {scalar}") from e
        self.validate_index(row_indx, 0)
        for col in range(self.cols):
            self._matrix[row_indx][col] = self._matrix[row_indx][col] * scalar

    def add_multiple_of_row(self, target_row: int, source_row: int, scalar: Any) -> None:
        """
        Add a multiple of one row to another row (ERO type 3).
        target_row += scalar * source_row
        
        :param target_row: Target row index
        :type target_row: int
        :param source_row: Source row index
        :type source_row: int
        :param scalar: Scalar to multiply the source row by
        :type scalar: Any
        """
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

    #TODO: cleanup method
    def elim(self, get_inv: bool = False, print_steps: bool = True, reduced: bool = True) -> 'Matrix':
        """
        Performs Gauss-Jordan elimination to compute RREF/REF or inverse matrix.
        
        :param get_inv: Whether to compute the inverse matrix instead of the REF/RREF.
        :type get_inv: bool
        :param print_steps: Whether to print each step of the elimination process.
        :type print_steps: bool
        :param reduced: Whether to compute RREF (True) or REF (False). Ignored if get_inv is True.
        :type reduced: bool
        :return: The RREF/REF matrix or the inverse matrix.
        :rtype: Matrix
        """
        if get_inv:
            reduced = True
        if get_inv and self.rows != self.cols:
            raise ValueError("Inverse can only be computed for square matrices.")

        mat = Matrix(self.matrix, self._field)

        inv: Matrix | None = None
        if get_inv:
            inv = Matrix.identity(self.rows, self.cols, self._field)

        def print_step(step_desc: str) -> None:
            if not print_steps:
                return
            print(step_desc)
            print("Matrix:")
            print(mat)
            if inv is not None:
                print("Inverse:")
                print(inv)
            print()

        pivot_row = 0
        for pivot_col in range(mat.cols):
            if pivot_row >= mat.rows:
                break

            row_with_pivot = mat.first_nonzero(pivot_col, pivot_row)
            if row_with_pivot == -1:
                continue

            #swap rows for nonzero pivot
            if row_with_pivot != pivot_row:
                mat.swap_rows(pivot_row, row_with_pivot)
                if inv is not None:
                    inv.swap_rows(pivot_row, row_with_pivot)
                print_step(f"R{pivot_row + 1} <-> R{row_with_pivot + 1}")

            #scale rows (only for reduced row echelon form)
            pivot_val = mat.matrix[pivot_row][pivot_col]
            if pivot_val != mat.field.one() and reduced:
                inv_pivot = pivot_val.mult_inv()
                mat.scale_row(pivot_row, inv_pivot)
                if inv is not None:
                    inv.scale_row(pivot_row, inv_pivot)
                print_step(f"R{pivot_row + 1} -> ({inv_pivot})R{pivot_row + 1}")

            #cancel elements in the pivot column
            if reduced:
                row_iter = range(mat.rows)
                pivot_inv: Any | None = None
            else:
                #don't elim elements above pivot row for REF
                row_iter = range(pivot_row + 1, mat.rows)
                pivot_inv = pivot_val.mult_inv()
            for r in row_iter:
                if r == pivot_row:
                    continue
                if reduced:
                    factor = -mat.matrix[r][pivot_col]
                else:
                    factor = -(mat.matrix[r][pivot_col] * pivot_inv)
                if factor == mat.field.zero():
                    continue
                mat.add_multiple_of_row(r, pivot_row, factor)
                if inv is not None:
                    inv.add_multiple_of_row(r, pivot_row, factor)
                print_step(f"R{r + 1} -> R{r + 1} + ({factor})R{pivot_row + 1}")
            print_step(f"Current Matrix: \n{mat}")
            if inv:
                print_step(f"current Inverse: \n{inv}")
            pivot_row += 1

        if inv is not None:
            #check if RREF was identity matrix (and thus the matrix was invertible)
            for i in range(mat.rows):
                for j in range(mat.cols):
                    expected = mat.field.one() if i == j else mat.field.zero()
                    if mat.matrix[i][j] != expected:
                        raise ValueError("Matrix is not invertible.")
            return inv

        return mat

    def det(self, print_steps: bool = True) -> Any:
        """
        Finds the determinant of the matrix using REF and diagonal multiplication.
        
        :param print_steps: Whether to output each step
        :type print_steps: bool
        :return: The determinant of the matrix
        :rtype: Any
        """
        if self.rows != self.cols:
            raise ValueError("Cannot compute determinant of non-square matrix.")

        mat = Matrix(self.matrix, self._field)
        det = self.field.one()
        # REF elimination with row swaps only (no scaling)
        pivot_row = 0
        for pivot_col in range(mat.cols):
            if pivot_row >= mat.rows:
                break

            row_with_pivot = mat.first_nonzero(pivot_col, pivot_row)
            if row_with_pivot == -1:
                continue

            #swapping rows
            if row_with_pivot != pivot_row:
                mat.swap_rows(pivot_row, row_with_pivot)
                det = -det
                if print_steps:
                    print(f"R{pivot_row + 1} <-> R{row_with_pivot + 1}; det *= -1 -> {det}")

            pivot_val = mat.matrix[pivot_row][pivot_col]
            pivot_inv = pivot_val.mult_inv()
            for r in range(pivot_row + 1, mat.rows):
                factor = -(mat.matrix[r][pivot_col] * pivot_inv)
                if factor == mat.field.zero():
                    continue
                mat.add_multiple_of_row(r, pivot_row, factor)
                if print_steps:
                    print(f"R{r + 1} -> R{r + 1} + ({factor})R{pivot_row + 1}; det={det}")
            if print_steps:
                print(f"Current Matrix: \n{mat}")
            pivot_row += 1
        # determinant is product of diagonal entries times swap sign
        for i in range(mat.rows):
            det = det * mat.matrix[i][i]
        return det

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
        """
        Return the field over which the matrix is defined.
        """
        return self._field

    def __str__(self) -> str:
        #individual elements need to have str() called to print properly
        result = "Matrix("
        for row in self._matrix:
            result += "[" + ", ".join(str(elem) for elem in row) +  "]\n"
        return result.strip() + ")"

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

    def __mul__(self, other: 'Matrix') -> 'Matrix':
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

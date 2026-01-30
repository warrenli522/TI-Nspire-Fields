try:
    from fractions import Fraction
    from abc import ABC, abstractmethod
except ImportError:  # MicroPython fallback
    class ABC:
        """Minimal ABC base class."""

    def abstractmethod(func):
        """Decorator to mark methods as abstract (no enforcement)."""
        return func

    def _gcd(a, b):
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a or 1

    class Fraction:
        """Simple rational number implementation (num/den) with reduction."""

        __slots__ = ("_num", "_den")

        def __init__(self, numerator=0, denominator=1):
            if denominator == 0:
                raise ZeroDivisionError("Denominator cannot be zero")

            num = numerator
            den = denominator

            if den < 0:
                num = -num
                den = -den

            g = _gcd(num, den)
            self._num = num // g
            self._den = den // g

        @property
        def numerator(self):
            return self._num

        @property
        def denominator(self):
            return self._den

        def __add__(self, other):
            other = Fraction(other) if isinstance(other, int) else other
            num = self._num * other._den + other._num * self._den
            den = self._den * other._den
            return Fraction(num, den)

        def __mul__(self, other):
            other = Fraction(other) if isinstance(other, int) else other
            return Fraction(self._num * other._num, self._den * other._den)

        def __truediv__(self, other):
            other = Fraction(other) if isinstance(other, int) else other
            if other._num == 0:
                raise ZeroDivisionError("division by zero")
            return Fraction(self._num * other._den, self._den * other._num)

        def __neg__(self):
            return Fraction(-self._num, self._den)

        def __eq__(self, other):
            if isinstance(other, int):
                return self._num == other * self._den
            elif isinstance(other, Fraction):
                return self._num == other._num and self._den == other._den
            else:
                return False

        def __bool__(self):
            return self._num != 0

        def __repr__(self):
            return "Fraction({0}, {1})".format(self._num, self._den)

        def __str__(self):
            if self._den == 1:
                return str(self._num)
            return "{0}/{1}".format(self._num, self._den)

try:
    import copy
except ImportError:  # MicroPython fallback
    class _CopyModule:
        @staticmethod
        def deepcopy(value):
            if isinstance(value, list):
                return [list(row) for row in value]
            return value
    copy = _CopyModule()

# TODO: refactor fields/field elements to be a single class (otherwise has issues w/ typing)
class FieldElement(ABC):
    """Abstract base class for field elements."""
    @abstractmethod
    def __init__(self, value):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    def __sub__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        return self + (-other)

    def __eq__(self, other):
        if other == 0:
            return self._value == self.base_field.zero()._value
        if other == 1:
            return self._value == self.base_field.one()._value
        return self.same_field(other) and self._value == other._value

    def __pow__(self,  b):
        if b == -1:
            return self.mult_inv()
        raise NotImplementedError("Only exponent -1 is implemented for field elements.")

    def __repr__(self):
        return "'{0}'".format(self._value)

    def __bool__(self):
        """Return True if the field element is not zero."""
        return self != self.base_field.zero()

    def __hash__(self):
        return hash((self.base_field, self._value))

    def same_field(self, other):
        """Check if two objects are elemnts from the same field."""
        return isinstance(other, FieldElement) and self.base_field == other.base_field

    def mult_inv(self):
        """Return the multiplicative inverse of the field element."""
        return self.base_field.one() / self

    # apparently abstract instance vars don't exist
    @property
    @abstractmethod
    def base_field(self):
        # TODO: determine if it should be a field instance or the class
        """Return the base field of the element."""

    @property
    @abstractmethod
    def _value(self):
        """Return the internal value of the field element."""

class Field(ABC):
    """Abstract base class for fields."""

    def __contains__(self, item):
        """Check if an item is an element of the field."""
        return isinstance(item, FieldElement) and isinstance(self, item.base_field)

    @classmethod
    @abstractmethod
    def zero(cls):
        """Return the additive identity element of the field."""

    @classmethod
    @abstractmethod
    def one(cls):
        """Return the multiplicative identity element of the field."""

    @property
    @abstractmethod
    def _field_elem_class(self):
        """Return underlying FieldElement class for use in instantiate()"""

    def instantiate(self, value):
        """Create a field element from a value."""
        if isinstance(value, self._field_elem_class):
            return value
        return self._field_elem_class(value)

class AnimalField(Field):
    """Custom animals field"""

    @classmethod
    def zero(cls):
        return AnimalElement("k")

    @classmethod
    def one(cls):
        return AnimalElement("c")

    @property
    def _field_elem_class(self):
        return AnimalElement

class AnimalElement(FieldElement):
    """Custom animals field element"""

    __value = ""  # for some reason instance variables can't override properties
    _symbols = ("k", "c", "d", "g")
    _addition_table = {
    ("g", "g"): "k", ("g", "d"): "c", ("g", "c"): "d", ("g", "k"): "g",
    ("d", "g"): "c", ("d", "d"): "k", ("d", "c"): "g", ("d", "k"): "d",
    ("c", "g"): "d", ("c", "d"): "g", ("c", "c"): "k", ("c", "k"): "c",
    ("k", "g"): "g", ("k", "d"): "d", ("k", "c"): "c", ("k", "k"): "k"
    }

    def __init__(self, value):
        if value not in self._symbols:
            raise ValueError("{0} is not a valid AnimalElement symbol.".format(value))
        self.__value = value
        self.__base_field = AnimalField


    _multiplication_table = {
    ("g", "g"): "d", ("g", "d"): "c", ("g", "c"): "g", ("g", "k"): "k",
    ("d", "g"): "c", ("d", "d"): "g", ("d", "c"): "d", ("d", "k"): "k",
    ("c", "g"): "g", ("c", "d"): "d", ("c", "c"): "c", ("c", "k"): "k",
    ("k", "g"): "k", ("k", "d"): "k", ("k", "c"): "k", ("k", "k"): "k"
    }

    @property
    def base_field(self):
        return self.__base_field

    @property
    def _value(self):
        return self.__value

    def __mul__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        result_symbol = self._multiplication_table[(self._value, other._value)]
        return AnimalElement(result_symbol)

    def __neg__(self):
        return self

    def __add__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        result_symbol = self._addition_table[(self._value, other._value)]
        return AnimalElement(result_symbol)

    def __truediv__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        if other == self.base_field.zero():
            raise ZeroDivisionError("Cannot divide {0} by {1}".format(self, other))
        mult_inv = ""
        for i in self._multiplication_table.items():
            if i[1] == "c" and i[0][0] == other._value:
                mult_inv = i[0][1]
        return self * AnimalElement(mult_inv)

class RationalField(Field):
    """Field of rational numbers using python's Fraction class."""

    @classmethod
    def zero(cls):
        return RationalNumber(Fraction(0, 1))

    @classmethod
    def one(cls):
        return RationalNumber(Fraction(1, 1))

    @property
    def _field_elem_class(self):
        return RationalNumber

class RationalNumber(FieldElement):
    """Represent rational numbers using python's Fraction class."""

    __value = Fraction(0, 1)

    def __init__(self, value):
        if isinstance(value, Fraction):
            self.__value = value
        elif isinstance(value, int):
            self.__value = Fraction(value)
        else:
            self.__value = Fraction(value[0], value[1])

    @property
    def _value(self):
        return self.__value

    @property
    def base_field(self):
        return RationalField

    def __add__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        return RationalNumber(self._value + other._value)

    def __mul__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        return RationalNumber(self._value * other._value)

    def __truediv__(self, other):
        if not self.same_field(other):
            raise TypeError(
                "Unsupported operand for items of type {0} and {1}!".format(type(self), type(other))
                )
        if other._value == 0:
            raise ZeroDivisionError("Cannot divide {0} by {1}".format(self, other))
        return RationalNumber(self._value / other._value)

    def __neg__(self):
        return RationalNumber(-self._value)

# TODO: make this a container class (need to change field classes first)
class Matrix:
    """A matrix modeled off of SymPy's DomainMatrix, but micropython compatible"""
    _matrix = None
    _field = None

    def __init__(self, matrix, field):
        self._field = field() if isinstance(field, type) else field

        for row in matrix:
            for idx, element in enumerate(row):
                # convert to field element (e.g. int to Fraction) and verify it is a field element
                element = self._field.instantiate(element)
                row[idx] = element

        self._matrix = matrix

    @classmethod
    def identity(cls, n, m, field):
        """Create an n x m identity matrix over the given field."""
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                if i == j:
                    row.append(field.one())
                else:
                    row.append(field.zero())
            result.append(row)
        return cls(result, field)

    def validate_index(self, row, col):
        """Validate that the given row and column indices are within the matrix bounds."""
        if row < 0 or row >= len(self._matrix):
            raise IndexError("Row index out of range. (input: {0})".format(row))
        if col < 0 or col >= len(self._matrix[0]):
            raise IndexError("Column index out of range. (input: {0})".format(col))

    def swap_rows(self, row1, row2):
        """
        Swap two rows of the matrix (ERO type 1).

        :param row1: First row index
        :param row2: Second row index
        """
        self.validate_index(row1, 0)
        self.validate_index(row2, 0)
        self._matrix[row1], self._matrix[row2] = self._matrix[row2], self._matrix[row1]

    def scale_row(self, row_indx, scalar):
        """
        Scale a row of the matrix (ERO type 2).

        :param row_indx: Row index to scale
        :param scalar: Scalar to multiply the row by
        """
        try:
            scalar = self._field.instantiate(scalar)
        except ValueError as e:
            raise ValueError("Invalid scalar for multiplication: {0}".format(scalar)) from e
        self.validate_index(row_indx, 0)
        for col in range(self.cols):
            self._matrix[row_indx][col] = self._matrix[row_indx][col] * scalar

    def add_multiple_of_row(self, target_row, source_row, scalar):
        """
        Add a multiple of one row to another row (ERO type 3).
        target_row += scalar * source_row

        :param target_row: Target row index
        :param source_row: Source row index
        :param scalar: Scalar to multiply the source row by
        """
        try:
            scalar = self._field.instantiate(scalar)
        except ValueError as e:
            raise ValueError("Invalid scalar for row addition: {0}".format(scalar)) from e
        self.validate_index(target_row, 0)
        self.validate_index(source_row, 0)
        for col in range(self.cols):
            product = self._matrix[source_row][col] * scalar
            self._matrix[target_row][col] = self._matrix[target_row][col] + product

    def first_nonzero(self, col, first_row=0):
        """Returns the index of the first nonzero element in a column, or -1 if all are zero."""
        self.validate_index(0, col)
        if first_row < 0 or first_row >= self.rows:
            raise IndexError("First row index out of range. (first_row: {0})".format(first_row))
        for row in range(first_row, self.rows):
            if self._matrix[row][col] != self._field.zero():
                return row
        return -1

    def get_pivot(self, col):
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

    # TODO: cleanup method
    def elim(self, get_inv=False, print_steps=True, reduced=True):
        """
        Performs Gauss-Jordan elimination to compute RREF/REF or inverse matrix.

        :param get_inv: Whether to compute the inverse matrix instead of the REF/RREF.
        :param print_steps: Whether to print each step of the elimination process.
        :param reduced: Whether to compute RREF (True) or REF (False). Ignored if get_inv is True.
        :return: The RREF/REF matrix or the inverse matrix.
        """
        if get_inv:
            reduced = True
        if get_inv and self.rows != self.cols:
            raise ValueError("Inverse can only be computed for square matrices.")

        mat = Matrix(self.matrix, self._field)

        inv = None
        if get_inv:
            inv = Matrix.identity(self.rows, self.cols, self._field)

        def print_step(step_desc):
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

            # swap rows for nonzero pivot
            if row_with_pivot != pivot_row:
                mat.swap_rows(pivot_row, row_with_pivot)
                if inv is not None:
                    inv.swap_rows(pivot_row, row_with_pivot)
                print_step("R{0} <-> R{1}".format(pivot_row + 1, row_with_pivot + 1))

            # scale rows (only for reduced row echelon form)
            pivot_val = mat.matrix[pivot_row][pivot_col]
            if pivot_val != mat.field.one() and reduced:
                inv_pivot = pivot_val.mult_inv()
                mat.scale_row(pivot_row, inv_pivot)
                if inv is not None:
                    inv.scale_row(pivot_row, inv_pivot)
                print_step("R{0} -> ({1})R{0}".format(pivot_row + 1, inv_pivot))

            # cancel elements in the pivot column
            if reduced:
                row_iter = range(mat.rows)
                pivot_inv = None
            else:
                # don't elim elements above pivot row for REF
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
                print_step("R{0} -> R{0} + ({1})R{2}".format(r + 1, factor, pivot_row + 1))
            print_step("Current Matrix: \n{0}".format(mat))
            if inv:
                print_step("current Inverse: \n{0}".format(inv))
            pivot_row += 1

        if inv is not None:
            # check if RREF was identity matrix (and thus the matrix was invertible)
            for i in range(mat.rows):
                for j in range(mat.cols):
                    expected = mat.field.one() if i == j else mat.field.zero()
                    if mat.matrix[i][j] != expected:
                        raise ValueError("Matrix is not invertible.")
            return inv

        return mat

    def det(self, print_steps=True):
        """
        Finds the determinant of the matrix using REF and diagonal multiplication.

        :param print_steps: Whether to output each step
        :return: The determinant of the matrix
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

            # swapping rows
            if row_with_pivot != pivot_row:
                mat.swap_rows(pivot_row, row_with_pivot)
                det = -det
                if print_steps:
                    print("R{0} <-> R{1}; det *= -1 -> {2}".format(pivot_row + 1, row_with_pivot + 1, det))

            pivot_val = mat.matrix[pivot_row][pivot_col]
            pivot_inv = pivot_val.mult_inv()
            for r in range(pivot_row + 1, mat.rows):
                factor = -(mat.matrix[r][pivot_col] * pivot_inv)
                if factor == mat.field.zero():
                    continue
                mat.add_multiple_of_row(r, pivot_row, factor)
                if print_steps:
                    print("R{0} -> R{0} + ({1})R{2}; det={3}".format(r + 1, factor, pivot_row + 1, det))
            if print_steps:
                print("Current Matrix: \n{0}".format(mat))
            pivot_row += 1
        # determinant is product of diagonal entries times swap sign
        for i in range(mat.rows):
            det = det * mat.matrix[i][i]
        return det

    @property
    def rows(self):
        """Number of rows in the matrix."""
        return len(self._matrix)

    @property
    def cols(self):
        """Number of columns in the matrix."""
        return len(self._matrix[0])

    @property
    def matrix(self):
        """Return a copy of the underlying matrix"""
        return self._matrix

    @property
    def field(self):
        """
        Return the field over which the matrix is defined.
        """
        return self._field

    def __str__(self):
        # individual elements need to have str() called to print properly
        result = "Matrix("
        for row in self._matrix:
            result += "[" + ", ".join(str(elem) for elem in row) +  "]\n"
        return result.strip() + ")"

    def __getitem__(self, key):
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

    def __setitem__(self, key, value):
        # don't want to allow direct access to underlying matrix to add validation

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
            if len(value) != self.cols:
                raise ValueError(
                    "Row length must be {0}. (input length: {1})".format(self.cols, len(value))
                    )
            new_row = []
            for element in value:
                try:
                    new_row.append(self._field.instantiate(element))
                except ValueError as e:
                    raise ValueError("Invalid field element: {0}".format(element)) from e
            self._matrix[index] = new_row
            return

        # Element assignment
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            self.validate_index(row, col)
            try:
                element = self._field.instantiate(value)
            except ValueError as e:
                raise ValueError("Invalid field element: {0}".format(value)) from e
            self._matrix[row][col] = element
            return

        raise IndexError("Invalid key type. Use m[row] or m[row, col].")

    def __mul__(self, other):
        """Matrix multiplication: self * other"""
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions for multiplication.")
        if not isinstance(other.field , type(self.field)):
            raise ValueError("Matrices must be over the same field for multiplication.")

        result_data = []
        for i in range(self.rows):
            result_row = []
            for j in range(other.cols):
                sum_element = self._field.zero()
                for k in range(self.cols):
                    product = self._matrix[i][k] * other[k, j]
                    sum_element = sum_element + product
                result_row.append(sum_element)
            result_data.append(result_row)

        return Matrix(result_data, self._field)

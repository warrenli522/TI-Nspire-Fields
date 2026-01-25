#typing: ignore
# =========================
# MicroPython compatibility
#NOTE: AI-generated micropython compatible version
# =========================

def abstractmethod(func):
    return func

class ABC:
    pass

# =========================
# Minimal Fraction class
# =========================

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

class Fraction:
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        g = _gcd(numerator, denominator)
        self.n = numerator // g
        self.d = denominator // g

    def __add__(self, other):
        return Fraction(self.n * other.d + other.n * self.d, self.d * other.d)

    def __sub__(self, other):
        return Fraction(self.n * other.d - other.n * self.d, self.d * other.d)

    def __mul__(self, other):
        return Fraction(self.n * other.n, self.d * other.d)

    def __truediv__(self, other):
        if other.n == 0:
            raise ZeroDivisionError("Division by zero")
        return Fraction(self.n * other.d, self.d * other.n)

    def __neg__(self):
        return Fraction(-self.n, self.d)

    def __eq__(self, other):
        return self.n == other.n and self.d == other.d

    def __repr__(self):
        return str(self.n) + "/" + str(self.d)

# =========================
# Field system
# =========================

class FieldElement(ABC):

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
        return self + (-other)

    def mult_inv(self):
        # Safe MicroPython-compatible multiplicative inverse
        return self.base_field.one() / self

    @property
    def base_field(self):
        # Return the field class for this element
        return type(self)._field_class

# =========================
# Field base
# =========================

class Field(ABC):

    _field_elem_class = None  # placeholder

    def __contains__(self, item):
        return isinstance(item, FieldElement) and isinstance(self, item.base_field)

    @classmethod
    @abstractmethod
    def zero(cls):
        pass

    @classmethod
    @abstractmethod
    def one(cls):
        pass

    @property
    @abstractmethod
    def _field_elem_class(self):
        return self._field_elem_class

    def instantiate(self, value):
        if isinstance(value, self._field_elem_class):
            return value
        return self._field_elem_class(value)

# =========================
# Animal field
# =========================

class AnimalField(Field):
    _field_elem_class = None  # will be set after AnimalElement definition

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
    _symbols = ("k", "c", "d", "g")
    _addition_table = {
        ("g", "g"): "k", ("g", "d"): "c", ("g", "c"): "d", ("g", "k"): "g",
        ("d", "g"): "c", ("d", "d"): "k", ("d", "c"): "g", ("d", "k"): "d",
        ("c", "g"): "d", ("c", "d"): "g", ("c", "c"): "k", ("c", "k"): "c",
        ("k", "g"): "g", ("k", "d"): "d", ("k", "c"): "c", ("k", "k"): "k"
    }
    _multiplication_table = {
        ("g", "g"): "c", ("g", "d"): "k", ("g", "c"): "g", ("g", "k"): "k",
        ("d", "g"): "k", ("d", "d"): "c", ("d", "c"): "d", ("d", "k"): "k",
        ("c", "g"): "g", ("c", "d"): "d", ("c", "c"): "c", ("c", "k"): "k",
        ("k", "g"): "k", ("k", "d"): "k", ("k", "c"): "k", ("k", "k"): "k"
    }
    _field_class = AnimalField

    def __init__(self, value):
        if value not in self._symbols:
            raise ValueError("Invalid symbol")
        self._value = value

    def __add__(self, other):
        return AnimalElement(self._addition_table[(self._value, other._value)])

    def __mul__(self, other):
        return AnimalElement(self._multiplication_table[(self._value, other._value)])

    def __truediv__(self, other):
        if other._value == "k":
            raise ZeroDivisionError("Division by zero")
        for (a, b), v in self._multiplication_table.items():
            if a == other._value and v == "c":
                return self * AnimalElement(b)

    def __neg__(self):
        return self

    def __repr__(self):
        return self._value

# =========================
# Rational field
# =========================

class RationalField(Field):
    _field_elem_class = None  # will be set after RationalNumber

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
    _field_class = RationalField

    def __init__(self, value):
        if isinstance(value, Fraction):
            self._value = value
        elif isinstance(value, int):
            self._value = Fraction(value, 1)
        elif isinstance(value, tuple):
            self._value = Fraction(value[0], value[1])
        else:
            raise ValueError("Invalid rational number")

    def __add__(self, other):
        return RationalNumber(self._value + other._value)

    def __mul__(self, other):
        return RationalNumber(self._value * other._value)

    def __truediv__(self, other):
        if other._value.n == 0:
            raise ZeroDivisionError("Division by zero")
        return RationalNumber(self._value / other._value)

    def __neg__(self):
        return RationalNumber(-self._value)

    def __repr__(self):
        return repr(self._value)

# =========================
# Matrix
# =========================

class Matrix:
    def __init__(self, matrix, field):
        self._field = field() if isinstance(field, type) else field
        self._matrix = []
        for row in matrix:
            self._matrix.append([self._field.instantiate(x) for x in row])

    def shape(self):
        return len(self._matrix), len(self._matrix[0])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
            return self._matrix[r][c]
        return self._matrix[key]

    def __setitem__(self, key, value):
        r, c = key
        self._matrix[r][c] = self._field.instantiate(value)

    def swap_rows(self, i, j):
        self._matrix[i], self._matrix[j] = self._matrix[j], self._matrix[i]

    def scale_row(self, i, scalar):
        self._matrix[i] = [scalar * x for x in self._matrix[i]]

    def add_rows(self, src, dst, scalar):
        self._matrix[dst] = [
            self._matrix[dst][j] + scalar * self._matrix[src][j]
            for j in range(len(self._matrix[0]))
        ]

    def __repr__(self):
        return "\n".join(str(row) for row in self._matrix)

# =========================
# Linear system solver
# =========================

def elim(matrix):
    rows, cols = matrix.shape()
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break

        if matrix[pivot_row, pivot_col]._value == matrix._field.zero()._value:
            for r in range(pivot_row + 1, rows):
                if matrix[r, pivot_col]._value != matrix._field.zero()._value:
                    matrix.swap_rows(pivot_row, r)
                    break
            else:
                continue

        pivot = matrix[pivot_row, pivot_col]
        matrix.scale_row(pivot_row, pivot.mult_inv())

        for r in range(rows):
            if r != pivot_row:
                factor = -matrix[r, pivot_col]
                matrix.add_rows(pivot_row, r, factor)

        pivot_row += 1

    return matrix

# =========================
# Set field class references
# =========================

AnimalField._field_elem_class = AnimalElement
RationalField._field_elem_class = RationalNumber

from typing import Any, Type

import sympy
from sympy import pretty, simplify #type: ignore
from sympy.polys.domains.field import Field as SympyField
from sympy.parsing.sympy_parser import parse_expr
from sympy import expand
from src.main.fields import Field, FieldElement

class SymbolicFieldElement(FieldElement):
    """Performs field operations on algebraic expressions using SymPy."""
    def __init__(self, symbol: str | sympy.Basic):
        self.__value: sympy.Basic = expand(simplify(parse_expr(str(symbol))))

    def __add__(self, other: FieldElement | sympy.Basic) -> FieldElement:
        if not isinstance(other, SymbolicFieldElement) and not isinstance(other, sympy.Basic):
            return NotImplemented
        return SymbolicFieldElement(str(self._value + other._value))
    def __mul__(self, other: FieldElement | sympy.Basic) -> FieldElement:
        if not isinstance(other, SymbolicFieldElement) and not isinstance(other, sympy.Basic):
            return NotImplemented
        return SymbolicFieldElement(str(self._value * other._value))
    def __neg__(self) -> FieldElement:
        return SymbolicFieldElement(str(-self._value))
    def __truediv__(self, other: FieldElement | sympy.Basic) -> FieldElement:
        if not isinstance(other, SymbolicFieldElement) and not isinstance(other, sympy.Basic):
            return NotImplemented
        if isinstance(other, sympy.Basic):
            return SymbolicFieldElement(str(self._value / other))
        return SymbolicFieldElement(str(self._value / other._value))

    @property
    def _value(self) -> sympy.Basic:
        return self.__value
    @property
    def base_field(self) -> Type[Field]:
        return SymbolicField
class SymbolicField(Field):
    """Custom field that uses SymPy for symbolic computation."""
    @classmethod
    def zero(cls) -> SymbolicFieldElement:
        return SymbolicFieldElement("0")
    @classmethod
    def one(cls) -> SymbolicFieldElement:
        return SymbolicFieldElement("1")
    @property
    def _field_elem_class(self) -> Type[FieldElement]:
        return SymbolicFieldElement

class SympyInterfaceField(SympyField): #pylint: disable=abstract-method
    """Make custom fields compatible w/ Sympy."""

    is_Field = True
    is_Exact = True

    def __init__(self, field_cls: Type[Field]):
        self.field = field_cls

    def __pow__(self, a: FieldElement, b: Any) -> FieldElement:
        if b == -1:
            return a.mult_inv()
        raise NotImplementedError("Only exponent -1 is implemented for field elements.")

    @property
    def zero(self):
        return self.field.zero()

    @property
    def one(self):
        return self.field.one()

    def __repr__(self) -> str:
        return f"SympyInterfaceField({self.field.__name__})"

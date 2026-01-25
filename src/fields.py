from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type
from fractions import Fraction

from sympy.polys.domains.field import Field as SympyField

class FieldElement(ABC):
    """Abstract base class for field elements."""

    @abstractmethod
    def __init__(self, value: Any):
        pass

    @abstractmethod
    def __add__(self, other: FieldElement) -> FieldElement:
        pass

    @abstractmethod
    def __mul__(self, other: FieldElement) -> FieldElement:
        pass

    @abstractmethod
    def __truediv__(self, other: FieldElement) -> FieldElement:
        pass

    @abstractmethod
    def __neg__(self) -> FieldElement:
        pass

    def __sub__(self, other: FieldElement) -> FieldElement:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        return self + (-other)

    def __eq__(self, other: Any) -> bool:
        if other == 0:
            return self._value == self.base_field.zero()._value
        if other == 1:
            return self._value == self.base_field.one()._value
        return self.same_field(other) and self._value == other._value

    def __pow__(self,  b: Any) -> FieldElement:
        if b == -1:
            return self.mult_inv()
        raise NotImplementedError("Only exponent -1 is implemented for field elements.")

    def __repr__(self) -> str:
        return f"'{self._value}'"

    def __bool__(self) -> bool:
        """Return True if the field element is not zero."""
        return self != self.base_field.zero()

    def __hash__(self) -> Any:
        return hash((self.base_field, self._value))

    def same_field(self, other: Any) -> bool:
        """Check if two objects are elemnts from the same field."""
        return isinstance(other, FieldElement) and self.base_field == other.base_field

    def mult_inv(self) -> FieldElement:
        """Return the multiplicative inverse of the field element."""
        return self.base_field.one() / self

    #apparently abstract instance vars don't exist
    @property
    @abstractmethod
    def base_field(self) -> Type[Field]:
        #TODO: determine if it should be a field instance or the class
        """Return the base field of the element."""

    @property
    @abstractmethod
    def _value(self) -> Any:
        """Return the internal value of the field element."""




class Field(ABC):
    """Abstract base class for fields."""

    def __contains__(self, item: Any) -> bool:
        """Check if an item is an element of the field."""
        return isinstance(item, FieldElement) and isinstance(self, item.base_field)


    @classmethod
    @abstractmethod
    def zero(cls) -> FieldElement:
        """Return the additive identity element of the field."""

    @classmethod
    @abstractmethod
    def one(cls) -> FieldElement:
        """Return the multiplicative identity element of the field."""

    @property
    @abstractmethod
    def _field_elem_class(self) -> Type[FieldElement]:
        """Return underlying FieldElement class for use in instantiate()"""

    def instantiate(self, value: Any) -> FieldElement:
        """Create a field element from a value."""
        if isinstance(value, self._field_elem_class):
            return value
        return self._field_elem_class(value)

class AnimalField(Field):
    """Custom animals field"""

    @classmethod
    def zero(cls) -> AnimalElement:
        return AnimalElement("k")

    @classmethod
    def one(cls) -> AnimalElement:
        return AnimalElement("c")

    @property
    def _field_elem_class(self) -> Type[FieldElement]:
        return AnimalElement

class AnimalElement(FieldElement):
    """Custom animals field element"""

    __value: str = "" #for some reason instance variables can't override properties
    _symbols: Tuple[str, ...] = ("k", "c", "d", "g")
    _addition_table: Dict[Tuple[str, str], str] = {
    ("g", "g"): "k", ("g", "d"): "c", ("g", "c"): "d", ("g", "k"): "g",
    ("d", "g"): "c", ("d", "d"): "k", ("d", "c"): "g", ("d", "k"): "d",
    ("c", "g"): "d", ("c", "d"): "g", ("c", "c"): "k", ("c", "k"): "c",
    ("k", "g"): "g", ("k", "d"): "d", ("k", "c"): "c", ("k", "k"): "k"
    }

    def __init__(self, value: str):
        if value not in self._symbols:
            raise ValueError(f"{value} is not a valid AnimalElement symbol.")
        self.__value = value
        self.__base_field = AnimalField


    _multiplication_table: Dict[Tuple[str, str], str] = {
    ("g", "g"): "d", ("g", "d"): "c", ("g", "c"): "g", ("g", "k"): "k",
    ("d", "g"): "c", ("d", "d"): "g", ("d", "c"): "d", ("d", "k"): "k",
    ("c", "g"): "g", ("c", "d"): "d", ("c", "c"): "c", ("c", "k"): "k",
    ("k", "g"): "k", ("k", "d"): "k", ("k", "c"): "k", ("k", "k"): "k"
    }

    @property
    def base_field(self) -> Type[Field]:
        return self.__base_field

    @property
    def _value(self) -> str:
        return self.__value

    def __mul__(self, other: FieldElement) -> AnimalElement:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        result_symbol = self._multiplication_table[(self._value, other._value)]
        return AnimalElement(result_symbol)

    def __neg__(self) -> AnimalElement:
        return self

    def __add__(self, other: FieldElement) -> AnimalElement:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        result_symbol = self._addition_table[(self._value, other._value)]
        return AnimalElement(result_symbol)

    def __truediv__(self, other: FieldElement) -> AnimalElement:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        if other == self.base_field.zero():
            raise ZeroDivisionError(f"Cannot divide {self} by {other}")
        mult_inv = ""
        for i in self._multiplication_table.items():
            if i[1] == "c" and i[0][0] == other._value:
                mult_inv = i[0][1]
        return self * AnimalElement(mult_inv)

class RationalField(Field):
    """Field of rational numbers using python's Fraction class."""

    @classmethod
    def zero(cls) -> RationalNumber:
        return RationalNumber(Fraction(0, 1))

    @classmethod
    def one(cls) -> RationalNumber:
        return RationalNumber(Fraction(1, 1))

    @property
    def _field_elem_class(self) -> Type[FieldElement]:
        return RationalNumber

class RationalNumber(FieldElement):
    """Represent rational numbers using python's Fraction class."""

    __value: Fraction = Fraction(0, 1)

    def __init__(self, value: int | Fraction | Tuple[int, int]):
        if isinstance(value, Fraction):
            self.__value = value
        elif isinstance(value, int):
            self.__value = Fraction(value)
        else:
            self.__value = Fraction(value[0], value[1])

    @property
    def _value(self) -> Fraction:
        return self.__value

    @property
    def base_field(self) -> Type[Field]:
        return RationalField

    def __add__(self, other: FieldElement) -> RationalNumber:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        return RationalNumber(self._value + other._value)

    def __mul__(self, other: FieldElement) -> RationalNumber:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        return RationalNumber(self._value * other._value)

    def __truediv__(self, other: FieldElement) -> RationalNumber:
        if not self.same_field(other):
            raise TypeError(
                f"Unsupported operand for items of type {type(self)} and {type(other)}!"
                )
        if other._value == 0:
            raise ZeroDivisionError(f"Cannot divide {self} by {other}")
        return RationalNumber(self._value / other._value)

    def __neg__(self) -> RationalNumber:
        return RationalNumber(-self._value)

class SympyInterfaceField(SympyField): #pylint: disable=abstract-method
    """Make fields defined here compatible w/ Sympy"""

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

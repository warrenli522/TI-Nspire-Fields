from typing import Any, Type

from sympy.polys.domains.field import Field as SympyField

from src.main.fields import Field, FieldElement

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

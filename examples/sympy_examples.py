from sympy.polys.matrices import DomainMatrix

from src.main.fields import AnimalElement, AnimalField
from src.sympy_functionality.sympy_field import SympyInterfaceField

#You can create a DomainMatrix over the custom fields
a = DomainMatrix([[AnimalElement("g"), AnimalElement("d"), AnimalElement("k")],
             [AnimalElement("c"), AnimalElement("k"), AnimalElement("d")],
             [AnimalElement("g"), AnimalElement("d"), AnimalElement("g")]],
             (3, 3), SympyInterfaceField(AnimalField)
             )

#sympy operations work as expected
print(a.inv())
print(a.inv() * a)
print(a.rref())
print(a.charpoly())
print(a.det()) #type: ignore
print(a.unify()) #type: ignore

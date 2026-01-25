from sympy.polys.matrices import DomainMatrix

from src.fields import AnimalElement, SympyInterfaceField, AnimalField

c = AnimalElement("c")
k = AnimalElement("k")
d = AnimalElement("d")
g = AnimalElement("g")

#sample usage
a = DomainMatrix([[AnimalElement("g"), AnimalElement("d"), AnimalElement("k")],
             [AnimalElement("c"), AnimalElement("k"), AnimalElement("d")],
             [AnimalElement("g"), AnimalElement("d"), AnimalElement("g")]],
             (3, 3), SympyInterfaceField(AnimalField)
             )
print(a.inv() * a)
print(a.charpoly())
print(a.det()) #type: ignore
print(a.unify()) #type: ignore

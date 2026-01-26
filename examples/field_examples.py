from src.main.fields import AnimalElement, AnimalField
from src.main.matrix import Matrix


m = Matrix([[AnimalElement("g"), AnimalElement("d"), AnimalElement("k")],
            [AnimalElement("c"), AnimalElement("k"), AnimalElement("d")],
            [AnimalElement("g"), AnimalElement("d"), AnimalElement("g")]], AnimalField)

print(m.elim())
print(m * m.elim(get_inv=True, print_steps=False))

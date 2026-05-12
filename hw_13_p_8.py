# from src.sympy_functionality.sympy_field import SymbolicField, SymbolicFieldElement, SympyInterfaceField
import random
from src.main.matrix import Matrix
from src.main.fields import RationalField, RationalNumber
from src.sympy_functionality.sympy_field import SymbolicField, SymbolicFieldElement
from sympy import init_printing, Matrix as SympyMatrix, QQ, ones, eye
import matplotlib as plt

init_printing()


points = []

for n in range(4, 5):
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(RationalNumber(i + 1) if i == j else RationalNumber(n))
        mat.append(row)
    mat = Matrix(mat, RationalField)
    mat.add_multiple_of_row(1, 2, RationalNumber(-1))
    mat.add_multiple_of_row(2, 3, RationalNumber(-1))
    print(mat)
    mat.add_multiple_of_row(3, 0, RationalNumber(-4))
    mat.add_multiple_of_row(3, 1, RationalNumber(-6))
    # mat.add_multiple_of_row(3, 2, RationalNumber(-18))
    print(mat)
    print(f"({n}, {int(str(mat.det(print_steps=False)).strip("'"))})")
    #print(f"\nDeterminant Value for n={n}: {mat.det(print_steps=False)}\n")

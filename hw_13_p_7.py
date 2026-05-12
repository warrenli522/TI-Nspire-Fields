# from src.sympy_functionality.sympy_field import SymbolicField, SymbolicFieldElement, SympyInterfaceField
import random
from src.main.matrix import Matrix
from src.main.fields import RationalField, RationalNumber
from src.sympy_functionality.sympy_field import SymbolicField, SymbolicFieldElement
from sympy import init_printing, Matrix as SympyMatrix, QQ, ones, eye
import matplotlib as plt

init_printing()


points = []

for n in range(1, 100):
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(SymbolicFieldElement("w+v") if i == j else SymbolicFieldElement("w"))
        mat.append(row)
    
    mat = Matrix(mat, SymbolicField)
    print(f"\nDeterminant Value for n={n}: {mat.det(print_steps=False)}\n")

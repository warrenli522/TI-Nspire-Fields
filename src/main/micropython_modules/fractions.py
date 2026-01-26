"""Minimal subset of Python's fractions module for MicroPython."""
from __future__ import annotations
from typing import Any

def _gcd(a: int, b: int) -> int:
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a or 1


class Fraction:
    """Simple rational number implementation (num/den) with reduction."""

    __slots__ = ("_num", "_den")

    def __init__(self, numerator: int = 0, denominator: int = 1):
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
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    def __add__(self, other: Fraction | int):
        other = Fraction(other) if isinstance(other, int) else other
        num = self._num * other._den + other._num * self._den
        den = self._den * other._den
        return Fraction(num, den)

    def __mul__(self, other: Fraction | int):
        other = Fraction(other) if isinstance(other, int) else other
        return Fraction(self._num * other._num, self._den * other._den)

    def __truediv__(self, other: Fraction | int):
        other = Fraction(other) if isinstance(other, int) else other
        if other._num == 0:
            raise ZeroDivisionError("division by zero")
        return Fraction(self._num * other._den, self._den * other._num)

    def __neg__(self):
        return Fraction(-self._num, self._den)

    def __eq__(self, other: Any):
        if isinstance(other, int):
            return self._num == other * self._den
        elif isinstance(other, Fraction):
            return self._num == other._num and self._den == other._den
        else:
            return False

    def __bool__(self):
        return self._num != 0

    def __repr__(self) -> str:
        return f"Fraction({self._num}, {self._den})"

    def __str__(self) -> str:
        if self._den == 1:
            return str(self._num)
        return f"{self._num}/{self._den}"

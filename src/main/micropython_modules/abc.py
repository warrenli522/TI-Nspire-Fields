"""Minimal subset of Python's abc module for MicroPython."""
from typing import Callable, Any


class ABC():
    """Minimal ABC base class."""

def abstractmethod(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark methods as abstract (no enforcement)."""
    func.__isabstractmethod__ = True #type: ignore[attr-defined]
    return func

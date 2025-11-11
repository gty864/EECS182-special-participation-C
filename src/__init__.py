# src/__init__.py
"""
q_sgd_momentum package.

A clean, reproducible demo of SGD vs Momentum on synthetic 2D data.
"""

# Optional: expose key classes at package level
from .trainers.base import SGDTrainer
from .data.synthetic import generate_data
from .models.logistic import LogisticRegression

__all__ = ["SGDTrainer", "generate_data", "LogisticRegression"]
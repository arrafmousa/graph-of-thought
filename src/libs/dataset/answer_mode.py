"""Supported answer extraction and comparison modes for math datasets."""
from __future__ import annotations

from enum import Enum


class AnswerMode(Enum):
    """How a dataset provider normalizes gold and generated answers."""

    NUMBER = "number"
    NUMBER_OR_TEXT = "number_or_text"
    MATH_TEXT = "math_text"
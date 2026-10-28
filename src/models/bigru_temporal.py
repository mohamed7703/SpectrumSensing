"""Bi-GRU Temporal Model Module Alias.

Exposes BiGRUModel from bigru_model for backward-compatibility with SKILL.md.
"""

from src.models.bigru_model import BiGRUModel

__all__ = ["BiGRUModel"]

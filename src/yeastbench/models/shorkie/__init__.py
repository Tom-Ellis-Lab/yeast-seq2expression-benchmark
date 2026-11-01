"""Shorkie network and benchmark inference wrapper."""
from yeastbench.models.shorkie.nn import ShorkieModule
from yeastbench.models.shorkie.wrapper import Shorkie

__all__ = ["Shorkie", "ShorkieModule"]

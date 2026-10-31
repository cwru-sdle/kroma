"""Compatibility wrapper around the inference strategies.

:data:`inference` is :func:`kroma.agents.inference.infer` under its original
name.
"""

from kroma.agents.inference import expand_term, infer

inference = infer

__all__ = ["expand_term", "inference"]

"""Per-method protocol adapters for the distributed scalability benchmark.

In the current implementation all five truth-discovery methods share the same
RPC shape (gather agent slices → coordinator aggregates → broadcast reliability),
so the adapters are thin policy objects rather than separate runners.
"""
from .runner import DistributedRunner, MethodProtocol, METHOD_PROTOCOLS

__all__ = ["DistributedRunner", "MethodProtocol", "METHOD_PROTOCOLS"]

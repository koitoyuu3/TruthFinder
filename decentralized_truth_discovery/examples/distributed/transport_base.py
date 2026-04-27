"""Abstract transport interfaces for the distributed scalability benchmark.

Two roles:
  * CoordinatorTransport — bound to a known endpoint, sends requests to one or
    more agent workers and gathers their replies.
  * AgentTransport — connects to the coordinator endpoint, receives requests,
    and pushes back replies.

The interfaces intentionally model a request/reply pattern (with optional
multi-round iteration handled at the call site) so that both single-pass methods
(SemanticTruth, SenteTruth, DecentTruth) and iterative ones (BasicTruth,
SenFeedTruth shock fallback) can share the same plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TransportStats:
    """Cumulative wire-level metrics collected on a single endpoint."""

    bytes_sent: int = 0
    bytes_recv: int = 0
    messages_sent: int = 0
    messages_recv: int = 0
    rpc_latencies_ms: List[float] = field(default_factory=list)

    def merge(self, other: "TransportStats") -> None:
        self.bytes_sent += other.bytes_sent
        self.bytes_recv += other.bytes_recv
        self.messages_sent += other.messages_sent
        self.messages_recv += other.messages_recv
        self.rpc_latencies_ms.extend(other.rpc_latencies_ms)

    def snapshot(self) -> Dict[str, Any]:
        latencies = sorted(self.rpc_latencies_ms)
        n = len(latencies)
        if n == 0:
            p50 = float("nan")
            p95 = float("nan")
        else:
            p50 = latencies[max(0, int(0.5 * (n - 1)))]
            p95 = latencies[max(0, int(0.95 * (n - 1)))]
        return {
            "bytes_sent": int(self.bytes_sent),
            "bytes_recv": int(self.bytes_recv),
            "messages_sent": int(self.messages_sent),
            "messages_recv": int(self.messages_recv),
            "rpc_count": n,
            "rpc_latency_p50_ms": float(p50),
            "rpc_latency_p95_ms": float(p95),
        }


class CoordinatorTransport:
    """Coordinator-side transport. One instance per coordinator process."""

    def __init__(self, endpoint: str, serialization: str = "msgpack"):
        self.endpoint = endpoint
        self.serialization = serialization
        self.stats = TransportStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, expected_agents: Sequence[str]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send(self, agent_id: str, payload: Any) -> None:
        """Send a request payload to a single agent (no wait)."""
        raise NotImplementedError

    def broadcast(self, agent_ids: Sequence[str], payload: Any) -> None:
        """Send the same payload to many agents (one-to-many)."""
        raise NotImplementedError

    def gather(
        self,
        agent_ids: Sequence[str],
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Block until a reply has been received from every listed agent."""
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.stats = TransportStats()


class AgentTransport:
    """Agent-side transport. One instance per worker process."""

    def __init__(self, endpoint: str, agent_id: str, serialization: str = "msgpack"):
        self.endpoint = endpoint
        self.agent_id = agent_id
        self.serialization = serialization
        self.stats = TransportStats()

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def recv(self, timeout_sec: Optional[float] = None) -> Any:
        """Block until a request from the coordinator arrives."""
        raise NotImplementedError

    def send(self, payload: Any) -> None:
        """Push a reply payload back to the coordinator."""
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.stats = TransportStats()

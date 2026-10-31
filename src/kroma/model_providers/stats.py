"""Running API totals (same fields as :class:`kroma.config.arguments.APIStats`)."""

from dataclasses import dataclass, fields

@dataclass
class APIStats:
    """Running cost and token totals for one model."""
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0
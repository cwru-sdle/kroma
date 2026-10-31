"""Dataclasses passed to the model wrappers."""

from dataclasses import dataclass

@dataclass
class APIStats():
    """Running cost and token totals for one model.

    Attributes:
        total_cost (float): Dollars across all calls.
        instance_cost (float): Dollars for the last call.
        tokens_sent (int): Prompt tokens across all calls.
        tokens_received (int): Completion tokens across all calls.
        api_calls (int): Calls made.
    """
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

@dataclass
class ModelArguments:
    """Settings for a chat model.

    Attributes:
        model_name (str): Provider model name.
        api_key (str | None): API key.
        temperature (float): Default sampling temperature.
        max_tokens (int): Default completion token limit.
        host_url (str | None): Base URL for a self-hosted endpoint.
    """
    model_name: str
    api_key: str = None
    temperature: float = 0.7
    max_tokens: int = 4000
    host_url: str = None

@dataclass
class ModelMetadata:
    """Per-token prices in dollars, from :data:`kroma.config.constants.METADATA`."""
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0

@dataclass
class EmbeddingArguments():
    """Settings for an embedding model.

    Attributes:
        model_name (str): Model name, such as a Hugging Face id.
        host_url (str | None): Base URL for a self-hosted endpoint.
    """
    model_name: str
    host_url: str = None
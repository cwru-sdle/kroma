"""Create the chat agents used for single-model and debate inference."""

from typing import Dict, Any, List
from kroma.inference.factory import create_inference_model

def create_agents_from_config(
    configs: List[Dict[str, str]]
) -> Dict[int, Any]:
    """Create one agent per config entry.

    Args:
        configs (list[dict[str, str]]): Entries with ``agent_type`` (a chat
            backend) and ``agent_name`` (a model), for example
            ``[{"agent_type": "openai", "agent_name": "gpt-4o-mini"}]``.

    Returns:
        dict[int, BaseModel]: Agents keyed by their position in ``configs``.
    """
    agents: Dict[int, Any] = {}
    for idx, cfg in enumerate(configs):
        agents[idx] = create_inference_model(cfg['agent_type'], cfg['agent_name'])
    return agents


def create_agents(
    n_agents: int,
    backend: str,
    model_name: str
) -> Dict[int, Any]:
    """Create ``n_agents`` agents with the same backend and model.

    Args:
        n_agents (int): Number of agents.
        backend (str): Chat backend name, such as ``"openai"``.
        model_name (str): Model name.

    Returns:
        dict[int, BaseModel]: Agents keyed ``0`` to ``n_agents - 1``.
    """
    return {
        i: create_inference_model(backend, model_name)
        for i in range(n_agents)
    }

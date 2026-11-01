"""Create chat and embedding models by backend name.

Backends are looked up in :data:`CHAT_BACKENDS` and :data:`EMBED_BACKENDS`;
add your own with :func:`register_chat_backend` and
:func:`register_embedding_backend`.
"""

from typing import Type, Dict
from kroma.utils.env import get_env
from kroma.config.constants import METADATA, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from kroma.config.arguments import ModelArguments, ModelMetadata, EmbeddingArguments

from kroma.model_providers.base_model import BaseModel, BaseEmbeddingModel
from kroma.model_providers.chat_providers import OpenAIProvider, TogetherProvider
from kroma.model_providers.embedding_providers import HFEmbeddingProvider, STEmbeddingProvider

# registry of available chat‐completion backends
CHAT_BACKENDS: Dict[str, Type] = {
    "openai": OpenAIProvider,
    "togetherai": TogetherProvider,
}

# registry of available embedding backends
EMBED_BACKENDS: Dict[str, Type] = {
    "huggingface": HFEmbeddingProvider,
    "sentence-transformers": STEmbeddingProvider,
}


def register_chat_backend(name: str, provider_cls: Type) -> None:
    """Register a chat backend.

    Args:
        name (str): Backend name, used for the API key variable
            ``<NAME>_API_KEY``.
        provider_cls (type): :class:`~kroma.model_providers.providers.ChatProvider`
            subclass taking ``api_key``.
    """
    CHAT_BACKENDS[name] = provider_cls


def register_embedding_backend(name: str, provider_cls: Type) -> None:
    """Register an embedding backend.

    Args:
        name (str): Backend name.
        provider_cls (type):
            :class:`~kroma.model_providers.providers.EmbeddingProvider`
            subclass taking ``model_name``.
    """
    EMBED_BACKENDS[name] = provider_cls


def create_inference_model(
    backend: str,
    model_name: str = DEFAULT_CHAT_MODEL,
) -> BaseModel:
    """Create a chat model for ``backend``.

    The API key is read from the environment variable
    ``<BACKEND>_API_KEY`` (for example ``OPENAI_API_KEY`` or
    ``TOGETHERAI_API_KEY``), and per-token prices from
    :data:`kroma.config.constants.METADATA`.

    Args:
        backend (str): Name in :data:`CHAT_BACKENDS`.
        model_name (str): Provider model name.

    Returns:
        BaseModel: The wrapped provider.

    Raises:
        ValueError: If the backend is not registered.
        Exception: If the API key variable is not set.
    """
    ProviderCls = CHAT_BACKENDS.get(backend)
    if ProviderCls is None:
        raise ValueError(f"unsupported chat backend: {backend}")

    # pull API key & build args/metadata
    api_key = get_env(f"{backend.upper()}_API_KEY")
    args = ModelArguments(model_name=model_name, api_key=api_key)

    meta_block = METADATA.get(backend, {})
    model_meta = meta_block.get(model_name, {})
    metadata = ModelMetadata(
        cost_per_input_token=model_meta.get("cost_per_input_token", 0.0),
        cost_per_output_token=model_meta.get("cost_per_output_token", 0.0),
    )

    # instantiate provider and wrap
    provider = ProviderCls(api_key=api_key)
    return BaseModel(provider, args, metadata)


def create_embedding_model(
    backend: str,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> BaseEmbeddingModel:
    """Create an embedding model for ``backend``.

    Args:
        backend (str): Name in :data:`EMBED_BACKENDS`, such as
            ``"huggingface"`` or ``"sentence-transformers"``.
        model_name (str): Model id, such as a Hugging Face model name.

    Returns:
        BaseEmbeddingModel: The wrapped provider.

    Raises:
        ValueError: If the backend is not registered.
    """
    ProviderCls = EMBED_BACKENDS.get(backend)
    if ProviderCls is None:
        raise ValueError(f"unsupported embedding backend: {backend}")
    args = EmbeddingArguments(model_name=model_name)
    provider = ProviderCls(model_name=model_name)
    return BaseEmbeddingModel(provider, args)


def generate_response(
    chat_backend: str,
    system_prompt: str,
    prompt: str,
    model_name: str = DEFAULT_CHAT_MODEL,
    include_tokens: bool = False,
    **kwargs,
):
    """Create a chat model and return its reply to one prompt.

    Args:
        chat_backend (str): Name in :data:`CHAT_BACKENDS`.
        system_prompt (str): System message.
        prompt (str): User message.
        model_name (str): Provider model name.
        include_tokens (bool): Also return token counts.
        **kwargs: Passed to :meth:`BaseModel.generate`.

    Returns:
        str | tuple: The reply, or ``(reply, input_tokens, output_tokens)``.
    """
    model = create_inference_model(chat_backend, model_name)
    return model.generate(
        system_prompt=system_prompt,
        prompt=prompt,
        include_tokens=include_tokens,
        **kwargs,
    )

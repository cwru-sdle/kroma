"""Provider-independent wrappers for chat and embedding models."""

from typing import Any, Dict, List, Optional
from kroma.config.arguments import (
    ModelArguments, 
    ModelMetadata, 
    EmbeddingArguments,
    APIStats,
)
from kroma.model_providers.providers import (
    ChatProvider, 
    EmbeddingProvider,
)
from kroma.config.constants import (
    CHARS_PER_TOKEN,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    TOTAL_TOKENS_LIMIT,
)
from kroma.decorators import retry

class BaseModel:
    """Chat model that adds retries, truncation and cost accounting to a provider.

    Create one with :func:`kroma.inference.factory.create_inference_model`.

    Args:
        provider (ChatProvider): Backend that sends requests.
        args (ModelArguments): Model name and default sampling settings.
        metadata (ModelMetadata): Per-token prices.

    Attributes:
        stats (APIStats): Running cost and token totals.
    """

    def __init__(
        self,
        provider: ChatProvider,
        args: ModelArguments,
        metadata: ModelMetadata,
    ):
        self.provider = provider
        self.args = args
        self.metadata = metadata
        self.stats = APIStats()  # your existing stats class

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        include_tokens: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Any = None,
        truncate: bool = True,
    ) -> Any:
        """Send one system and user message and return the reply.

        Transient provider errors are retried (see
        :func:`kroma.decorators.retry`).

        Args:
            system_prompt (str): System message.
            prompt (str): User message.
            include_tokens (bool): Return token counts and update
                :attr:`stats`.
            temperature (float | None): Overrides the default temperature.
            max_tokens (int | None): Overrides the default completion limit.
            response_format: Structured-output schema, passed to the provider.
            truncate (bool): Drop older messages that do not fit the token
                budget.

        Returns:
            str | tuple: The reply, or ``(reply, input_tokens, output_tokens)``
            when ``include_tokens`` is true.
        """
        # 1. construct message list
        messages: List[Dict[str,str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        # 2. truncate if needed
        if truncate:
            messages = self._truncate_conversation(messages)

        # 3. pick config
        temp = temperature if temperature is not None else self.args.temperature
        mtok = max_tokens if max_tokens is not None else self.args.max_tokens

        # 4. send to provider, retrying transient API failures
        raw = self._complete(messages, temp, mtok, response_format)

        # 5. parse response
        msg, in_tok, out_tok = self.provider.parse_response(raw, include_tokens)

        # 6. update stats (if we know tokens)
        if include_tokens and in_tok is not None and out_tok is not None:
            cost = (
                in_tok  * self.metadata.cost_per_input_token +
                out_tok * self.metadata.cost_per_output_token
            )
            self.stats.total_cost      += cost
            self.stats.instance_cost   = cost
            self.stats.tokens_sent    += in_tok
            self.stats.tokens_received+= out_tok
            self.stats.api_calls      += 1

        return (msg, in_tok, out_tok) if include_tokens else msg

    @retry(attempts=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF_SECONDS)
    def _complete(self, messages, temp, mtok, response_format):
        return self.provider.create_completion(
            model=self.args.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=mtok,
            response_format=response_format,
        )

    def _truncate_conversation(self, messages: List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Keep the system prompt plus as many most recent messages
        as fit under TOTAL_TOKENS_LIMIT - max_tokens.
        """
        # quick token estimate (1 token ~4 chars)
        def est_tokens(text: str) -> int:
            return len(text) // CHARS_PER_TOKEN

        keep = [messages[0]]
        used = est_tokens(messages[0]["content"])
        allowed = TOTAL_TOKENS_LIMIT - self.args.max_tokens

        # walk from the end backwards
        for msg in reversed(messages[1:]):
            tok = est_tokens(msg["content"])
            if used + tok <= allowed:
                keep.append(msg)
                used += tok
        keep.reverse()
        if len(keep) < len(messages):
            # insert a notice
            keep.insert(1, {
                "role": "system",
                "content": "⚠️ previous messages truncated due to length."
            })
        return keep

class BaseEmbeddingModel:
    """Embedding model with a simple :meth:`encode` method.

    Create one with :func:`kroma.inference.factory.create_embedding_model`.

    Args:
        provider (EmbeddingProvider): Backend that computes vectors.
        args (EmbeddingArguments): Model settings.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        args: EmbeddingArguments,
    ):
        self.provider = provider
        self.args = args

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Embed each string.

        Args:
            texts (list[str]): Strings to embed.

        Returns:
            list[list[float]]: One vector per string.
        """
        return self.provider.embed(texts)

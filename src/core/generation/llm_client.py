"""
LLM Client for RAG Pipeline.

Simple OpenAI client with retries and streaming support.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float


class LLMClient:
    """
    Simple OpenAI LLM client.

    Usage:
        client = LLMClient(model="gpt-4o-mini")
        response = await client.generate(prompt, system_prompt)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Args:
            model: Model name (e.g. "gpt-4o-mini", "openai/gpt-4o-mini")
            api_key: API key (uses OPENAI_API_KEY env if not provided)
            base_url: Base URL for OpenAI-compatible API (OpenRouter, local, etc.)
            max_retries: Max retry attempts
            timeout: Request timeout in seconds
        """
        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Generate response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content, model, usage, latency
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=self.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM one chunk at a time.

        Yields:
            Content delta strings as they arrive.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def close(self):
        """Close client."""
        await self._client.close()


def create_llm_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> LLMClient:
    """Create LLM client. Works with any OpenAI-compatible API."""
    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )

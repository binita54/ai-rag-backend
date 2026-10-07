"""Redis-backed conversation memory."""

import json

import redis.asyncio as redis_asyncio
from redis.exceptions import RedisError

from app.core.config import get_settings


class MemoryStoreError(Exception):
    """Base exception for memory store failures."""


class RedisMemoryStore:
    """Redis-backed conversation memory store."""

    def __init__(
        self,
        client: redis_asyncio.Redis | None = None,
        redis_url: str | None = None,
        max_history: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or redis_asyncio.from_url(
            redis_url or settings.REDIS_URL,
            decode_responses=True,
        )
        self._max_history = (
            settings.REDIS_MAX_HISTORY if max_history is None else max_history
        )
        self._ttl_seconds = (
            settings.REDIS_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        )
        if self._max_history < 1:
            raise ValueError("max_history must be at least 1")

    def _key(self, conversation_id: str) -> str:
        return f"chat:history:{conversation_id}"

    async def append_message(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """Append a message and trim history to the configured maximum."""
        key = self._key(conversation_id)
        message = json.dumps({"role": role, "content": content})
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, message)
                pipe.ltrim(key, -self._max_history, -1)
                pipe.expire(key, self._ttl_seconds)
                await pipe.execute()
        except RedisError as error:
            raise MemoryStoreError(
                f"Failed to append message to conversation "
                f"'{conversation_id}': {error}"
            ) from error

    async def get_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Return conversation history in chronological order."""
        key = self._key(conversation_id)
        try:
            if limit is not None and limit > 0:
                raw_messages = await self._client.lrange(key, -limit, -1)
            else:
                raw_messages = await self._client.lrange(key, 0, -1)
        except RedisError as error:
            raise MemoryStoreError(
                f"Failed to read history for conversation "
                f"'{conversation_id}': {error}"
            ) from error

        messages: list[dict[str, str]] = []
        for raw_message in raw_messages:
            try:
                parsed = json.loads(raw_message)
                messages.append(
                    {
                        "role": str(parsed["role"]),
                        "content": str(parsed["content"]),
                    }
                )
            except (TypeError, ValueError, KeyError) as error:
                raise MemoryStoreError(
                    f"Corrupted message in conversation '{conversation_id}'"
                ) from error
        return messages

    async def clear(self, conversation_id: str) -> None:
        """Delete all history for a conversation."""
        try:
            await self._client.delete(self._key(conversation_id))
        except RedisError as error:
            raise MemoryStoreError(
                f"Failed to clear conversation '{conversation_id}': {error}"
            ) from error

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.aclose()

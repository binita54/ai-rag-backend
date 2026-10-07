"""Tests for the Redis conversation memory store."""

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from app.services.memory import MemoryStoreError, RedisMemoryStore


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def memory_store(redis_client) -> RedisMemoryStore:
    return RedisMemoryStore(
        client=redis_client, max_history=5, ttl_seconds=3600
    )


async def test_append_and_retrieve_single_message(
    memory_store: RedisMemoryStore,
) -> None:
    await memory_store.append_message("conv1", "user", "Hello")
    history = await memory_store.get_history("conv1")
    assert history == [{"role": "user", "content": "Hello"}]


async def test_chronological_order_preserved(
    memory_store: RedisMemoryStore,
) -> None:
    await memory_store.append_message("conv1", "user", "first")
    await memory_store.append_message("conv1", "assistant", "second")
    await memory_store.append_message("conv1", "user", "third")

    history = await memory_store.get_history("conv1")
    assert [message["content"] for message in history] == [
        "first",
        "second",
        "third",
    ]


async def test_user_and_assistant_messages(
    memory_store: RedisMemoryStore,
) -> None:
    await memory_store.append_message("conv1", "user", "question")
    await memory_store.append_message("conv1", "assistant", "answer")

    history = await memory_store.get_history("conv1")
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


async def test_max_history_trimming(redis_client) -> None:
    store = RedisMemoryStore(
        client=redis_client, max_history=3, ttl_seconds=3600
    )
    for index in range(5):
        await store.append_message("conv1", "user", f"message{index}")

    history = await store.get_history("conv1")
    assert [message["content"] for message in history] == [
        "message2",
        "message3",
        "message4",
    ]


async def test_empty_conversation_returns_empty_list(
    memory_store: RedisMemoryStore,
) -> None:
    assert await memory_store.get_history("nonexistent") == []


async def test_clear_conversation(
    memory_store: RedisMemoryStore,
) -> None:
    await memory_store.append_message("conv1", "user", "Hello")
    await memory_store.clear("conv1")
    assert await memory_store.get_history("conv1") == []


async def test_multiple_conversations_isolated(
    memory_store: RedisMemoryStore,
) -> None:
    await memory_store.append_message("conv1", "user", "conv1 message")
    await memory_store.append_message("conv2", "user", "conv2 message")

    history1 = await memory_store.get_history("conv1")
    history2 = await memory_store.get_history("conv2")
    assert history1 == [{"role": "user", "content": "conv1 message"}]
    assert history2 == [{"role": "user", "content": "conv2 message"}]


async def test_history_limit_returns_recent_messages(
    memory_store: RedisMemoryStore,
) -> None:
    for index in range(4):
        await memory_store.append_message("conv1", "user", f"message{index}")

    history = await memory_store.get_history("conv1", limit=2)
    assert [message["content"] for message in history] == [
        "message2",
        "message3",
    ]


async def test_ttl_applied_to_key(
    memory_store: RedisMemoryStore, redis_client
) -> None:
    await memory_store.append_message("conv1", "user", "Hello")
    ttl = await redis_client.ttl("chat:history:conv1")
    assert 0 < ttl <= 3600


async def test_redis_errors_are_wrapped(
    memory_store: RedisMemoryStore, monkeypatch
) -> None:
    async def failing_lrange(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(memory_store._client, "lrange", failing_lrange)
    with pytest.raises(MemoryStoreError):
        await memory_store.get_history("conv1")


async def test_close_closes_connection(
    memory_store: RedisMemoryStore, redis_client
) -> None:
    assert memory_store._client is redis_client
    await memory_store.close()

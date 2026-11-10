"""Doubles of the repository container and the transaction for the store and update transaction tests."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from tests.tools._generation_jobs import InMemoryGenerationJobs


def transaction_repos(jobs: InMemoryGenerationJobs, entry_id: str) -> MagicMock:
    """Return repositories whose writes succeed, holding their generation jobs in ``jobs``.

    A store inserts ``entry_id`` as a new entry, an update finds it writable, and the entry
    holds its embeddings and its summary, until a test says otherwise.

    Args:
        jobs: The job repository double.
        entry_id: The id of the entry the writes target.

    Returns:
        The repository container double.
    """
    repos = MagicMock()
    repos.context.store_with_deduplication = AsyncMock(return_value=(entry_id, False))
    repos.context.insert_entry = AsyncMock(return_value=entry_id)
    repos.context.has_summary = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.touch_updated_at = AsyncMock(return_value=True)
    repos.context.entry_exists = AsyncMock(return_value=True)
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)
    repos.grants.store_group_read_grants = AsyncMock()
    repos.embeddings.exists = AsyncMock(return_value=True)
    repos.embeddings.store_chunked = AsyncMock()
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=False)
    repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
    repos.index_nodes.replace_nodes_for_context = AsyncMock()
    repos.generation_jobs = jobs
    return repos


def sqlite_txn() -> MagicMock:
    """Return a transaction double on SQLite."""
    txn = MagicMock()
    txn.backend_type = 'sqlite'
    return txn


def record_calls(
    monkeypatch: pytest.MonkeyPatch, target: object, name: str, order: list[str], label: str,
) -> None:
    """Record ``label`` in ``order`` whenever ``target.name`` is awaited, letting the call through."""
    method = getattr(target, name)

    async def _recording(*args: object, **kwargs: object) -> object:
        order.append(label)
        return await method(*args, **kwargs)

    monkeypatch.setattr(target, name, _recording)

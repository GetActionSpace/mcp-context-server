"""Tests for ContextRepository by-id reads and the existence, content-type and summary probes."""

import re
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.backends.base import TransactionContext
from app.backends.base import base_relation
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositoryGetById:
    """Test get_by_ids operations."""

    @pytest.mark.asyncio
    async def test_get_by_ids_single(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting single entry by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='get_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows = await repos.context.get_by_ids([ctx_id], scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_multiple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting multiple entries by IDs."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='multi_get',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )
            ids.append(ctx_id)

        rows = await repos.context.get_by_ids(ids, scope=LOCAL_SCOPE)

        assert len(rows) == 3
        returned_ids = {r['id'] for r in rows}
        assert returned_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting entries with empty ID list."""
        rows = await repos.context.get_by_ids([], scope=LOCAL_SCOPE)

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting nonexistent IDs returns empty."""
        rows = await repos.context.get_by_ids([generate_id(), generate_id()], scope=LOCAL_SCOPE)

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_partial_match(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='partial_get',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        rows = await repos.context.get_by_ids([ctx_id, generate_id()], scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_spans_multiple_chunks_preserves_order(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """get_by_ids chunks an oversized id list and preserves the ordering contract.

        A 33,000-id list bound as one IN clause exceeds SQLite's default 32,766
        per-statement variable ceiling ("too many SQL variables"), so the fetch is
        issued in bounded chunks like delete_by_ids. Real ids are planted on both
        sides of the 900-id chunk boundary with the OLDEST rows in the FIRST chunk
        and the NEWEST row in the LAST chunk, so plain per-chunk concatenation
        would invert the order -- the assertion therefore also proves the
        accumulated rows are re-sorted into the single-statement
        ORDER BY created_at DESC, id DESC contract.
        """
        real_ids: list[str] = []
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='chunk_get_thread',
                source='user',
                content_type='text',
                text_content=f'Chunk get entry {i}',
            )
            real_ids.append(ctx_id)

        # Expected order = the documented contract, computed from per-row reads
        # (each a single-chunk fetch): created_at DESC, id DESC.
        keyed: list[tuple[str, str]] = []
        for ctx_id in real_ids:
            row = (await repos.context.get_by_ids([ctx_id], scope=LOCAL_SCOPE))[0]
            keyed.append((row['created_at'], row['id']))
        expected_ids = [entry_id for _created_at, entry_id in sorted(keyed, reverse=True)]

        # Non-existent ids pad the list past the statement variable ceiling; the
        # oldest real rows go into the first chunk and the newest into the last.
        mixed = [generate_id() for _ in range(33000)]
        oldest_first = list(reversed(expected_ids))
        for pos, real_id in zip((0, 450, 899, 900, 32999), oldest_first, strict=True):
            mixed[pos] = real_id

        rows = await repos.context.get_by_ids(mixed, scope=LOCAL_SCOPE)

        assert [row['id'] for row in rows] == expected_ids


class TestGetByIdsReadBack:
    """get_by_ids reports each row's visibility and whether the caller owns it, never the owner itself."""

    @pytest.mark.asyncio
    async def test_rows_carry_visibility_and_ownership(self, repos: RepositoryContainer) -> None:
        """The caller's own private row reads back as owned, another principal's public row as not owned."""
        own_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE, visibility='private', thread_id='read_back', source='user',
            content_type='text', text_content='own private entry',
        )
        foreign_id, _ = await repos.context.store_with_deduplication(
            scope=AccessScope('alice', frozenset()), visibility='public', thread_id='read_back', source='agent',
            content_type='text', text_content='alice public entry',
        )

        rows = [dict(row) for row in await repos.context.get_by_ids([own_id, foreign_id], scope=LOCAL_SCOPE)]

        assert {row['id']: (row['visibility'], row['is_owner']) for row in rows} == {
            own_id: ('private', 1), foreign_id: ('public', 0),
        }
        assert all('owner_id' not in row for row in rows)

    @pytest.mark.asyncio
    async def test_system_scope_owns_every_row(self, repos: RepositoryContainer) -> None:
        """The system scope's empty predicate admits every row as owned."""
        foreign_id, _ = await repos.context.store_with_deduplication(
            scope=AccessScope('alice', frozenset()), visibility='private', thread_id='read_back', source='agent',
            content_type='text', text_content='alice private entry',
        )

        [row] = await repos.context.get_by_ids([foreign_id], scope=SYSTEM_SCOPE)

        assert (row['visibility'], row['is_owner']) == ('private', 1)

    @pytest.mark.asyncio
    async def test_postgresql_binds_the_owner_first(self) -> None:
        """On PostgreSQL the ownership bind of the select list is $1, the ids follow and the read predicate comes last."""
        statements: list[tuple[str, tuple[object, ...]]] = []

        class _RecordingConnection:
            async def fetch(self, query: str, *args: object) -> list[object]:
                statements.append((query, args))
                return []

        async def _execute_read(closure: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
            _ = access_scope
            return await closure(_RecordingConnection())

        pg_backend = Mock()
        pg_backend.backend_type = 'postgresql'
        pg_backend.readable_relation = base_relation
        pg_backend.execute_read = _execute_read
        repo = ContextRepository(cast(StorageBackend, pg_backend))

        await repo.get_by_ids(['a' * 32, 'b' * 32], scope=AccessScope('bob', frozenset({'team-x'})))

        [(query, args)] = statements
        flat = ' '.join(query.split())
        assert sorted({int(number) for number in re.findall(r'\$(\d+)', query)}) == list(range(1, len(args) + 1))
        assert 'summary, created_at, updated_at, visibility, CASE WHEN context_entries.owner_id = $1 THEN 1' in flat
        assert ' AS is_owner FROM context_entries WHERE id IN ($2,$3) AND (context_entries.owner_id = $4 ' in flat
        assert args == ('bob', 'a' * 32, 'b' * 32, 'bob', 'bob', ['team-x'])


async def _store_local_entries(repos: RepositoryContainer, count: int, thread_id: str) -> list[str]:
    """Store ``count`` entries owned by the default principal and return their ids."""
    stored: list[str] = []
    for i in range(count):
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id=thread_id,
            source='user',
            content_type='text',
            text_content=f'{thread_id} entry {i}',
        )
        stored.append(ctx_id)
    return stored


def _pad_to_1000_ids(real_ids: list[str]) -> list[str]:
    """Return 1,000 ids with the real ids at the first, last-of-chunk and last positions."""
    ids = [generate_id() for _ in range(1000)]
    for position, real_id in zip((0, 899, 999), real_ids, strict=True):
        ids[position] = real_id
    return ids


@pytest.mark.usefixtures('sqlite_999_variables')
class TestScopedReadsUnderVariableCap:
    """The scoped id reads fit each 900-id chunk plus its access binds within 999 variables."""

    @pytest.mark.asyncio
    async def test_get_by_ids_binds_1000_ids_with_scope(self, repos: RepositoryContainer) -> None:
        """A 1,000-id read under a scope binds without error and returns every stored row."""
        real_ids = await _store_local_entries(repos, 3, 'capped_get')

        rows = await repos.context.get_by_ids(_pad_to_1000_ids(real_ids), scope=LOCAL_SCOPE)

        assert {row['id']: (row['visibility'], row['is_owner']) for row in rows} == dict.fromkeys(real_ids, ('private', 1))

    @pytest.mark.asyncio
    async def test_probe_ids_binds_1000_ids_with_scope(self, repos: RepositoryContainer) -> None:
        """A 1,000-id access probe binds without error and reports every stored row."""
        real_ids = await _store_local_entries(repos, 3, 'capped_probe')

        access = await repos.context.probe_ids(_pad_to_1000_ids(real_ids), scope=LOCAL_SCOPE)

        assert set(access) == set(real_ids)
        assert all(entry.can_write and entry.is_owner for entry in access.values())

    @pytest.mark.asyncio
    async def test_probe_ids_of_an_empty_list_is_empty(self, repos: RepositoryContainer) -> None:
        """No ids means no statement and an empty result."""
        assert await repos.context.probe_ids([], scope=LOCAL_SCOPE) == {}


class TestContextRepositoryUpdate:
    """Test update operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test checking if entry exists."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        probe = await repos.context.check_entry_exists(ctx_id, scope=LOCAL_SCOPE)
        assert probe.exists is True
        assert probe.source == 'user'
        assert isinstance(probe.version, int)
        assert probe.version == 0
        assert probe.owner_id == 'local'
        assert probe.can_write is True

        missing = await repos.context.check_entry_exists(generate_id(), scope=LOCAL_SCOPE)
        assert missing.exists is False
        assert missing.source is None
        assert missing.version is None
        assert missing.owner_id is None
        assert missing.can_write is False

    @pytest.mark.asyncio
    async def test_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """entry_exists returns True for a stored id and False for an absent one."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='entry_exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        assert await repos.context.entry_exists(ctx_id, scope=LOCAL_SCOPE) is True
        assert await repos.context.entry_exists(generate_id(), scope=LOCAL_SCOPE) is False

    @pytest.mark.asyncio
    async def test_entry_exists_locks_parent_row_on_postgresql_transaction(self) -> None:
        """On PostgreSQL the in-transaction presence check locks the parent row.

        The tags-only / images-only update guard runs entry_exists on the open
        transaction connection; it must emit FOR KEY SHARE OF context_entries so a
        concurrent DELETE blocks until commit and cannot leave the child tag/image
        writes violating the foreign key; naming context_entries confines the lock to
        that parent row. Outside a transaction the lock would release at statement end,
        so it must NOT be emitted there. Both statements carry the write predicate
        after the id. Both cases are asserted against a recording connection without
        needing a live PostgreSQL.
        """
        txn_conn = AsyncMock()
        txn_conn.fetchrow = AsyncMock(return_value={'?column?': 1})
        txn_backend = Mock()
        txn_backend.backend_type = 'postgresql'
        txn = Mock()
        txn.backend_type = 'postgresql'
        txn.connection = txn_conn

        repo_txn = ContextRepository(cast(StorageBackend, txn_backend))
        assert await repo_txn.entry_exists('abc123', scope=LOCAL_SCOPE, txn=cast(TransactionContext, txn)) is True
        txn_sql = txn_conn.fetchrow.call_args.args[0]
        assert txn_sql.endswith(' LIMIT 1 FOR KEY SHARE OF context_entries')
        assert 'WHERE id = $1 AND (context_entries.owner_id = $2 OR EXISTS' in txn_sql
        assert txn_conn.fetchrow.call_args.args[1:] == ('abc123', 'local', 'local', [])

        pool_conn = AsyncMock()
        pool_conn.fetchrow = AsyncMock(return_value={'?column?': 1})

        async def _execute_read(closure: Callable[[object], Awaitable[bool]], *, access_scope: BackendAccess) -> bool:
            _ = access_scope
            return await closure(pool_conn)

        pool_backend = Mock()
        pool_backend.backend_type = 'postgresql'
        pool_backend.execute_read = _execute_read

        repo_pool = ContextRepository(cast(StorageBackend, pool_backend))
        assert await repo_pool.entry_exists('abc123', scope=LOCAL_SCOPE) is True
        pool_sql = pool_conn.fetchrow.call_args.args[0]
        assert 'FOR KEY SHARE' not in pool_sql
        assert 'WHERE id = $1 AND (context_entries.owner_id = $2 OR EXISTS' in pool_sql

    @pytest.mark.asyncio
    async def test_get_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text content',
        )

        content_type = await repos.context.get_content_type(ctx_id, scope=LOCAL_SCOPE)

        assert content_type == 'text'

    @pytest.mark.asyncio
    async def test_get_content_type_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type for nonexistent entry."""
        content_type = await repos.context.get_content_type(generate_id(), scope=LOCAL_SCOPE)

        assert content_type is None

    @pytest.mark.asyncio
    async def test_update_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test updating content type."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='update_type',
            source='user',
            content_type='text',
            text_content='Content',
        )

        await repos.context.update_content_type(ctx_id, 'multimodal', scope=LOCAL_SCOPE)

        new_type = await repos.context.get_content_type(ctx_id, scope=LOCAL_SCOPE)
        assert new_type == 'multimodal'


class TestHasSummary:
    """has_summary reports whether an entry the scope may modify holds a summary."""

    @pytest.mark.asyncio
    async def test_entry_with_a_summary(self, repos: RepositoryContainer) -> None:
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE, visibility='private', thread_id='summary_thread', source='user',
            content_type='text', text_content='Summarized text', summary='A summary.',
        )

        assert await repos.context.has_summary(ctx_id, scope=LOCAL_SCOPE) is True

    @pytest.mark.asyncio
    async def test_entry_without_a_summary(self, repos: RepositoryContainer) -> None:
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE, visibility='private', thread_id='summary_thread', source='user',
            content_type='text', text_content='Unsummarized text',
        )

        assert await repos.context.has_summary(ctx_id, scope=LOCAL_SCOPE) is False

    @pytest.mark.asyncio
    async def test_absent_entry(self, repos: RepositoryContainer) -> None:
        assert await repos.context.has_summary(generate_id(), scope=LOCAL_SCOPE) is None

    @pytest.mark.asyncio
    async def test_entry_the_scope_may_only_read(self, repos: RepositoryContainer) -> None:
        """Another principal's public entry is readable but not modifiable, so it reads as absent."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=AccessScope('alice', frozenset()), visibility='public', thread_id='summary_thread',
            source='user', content_type='text', text_content='Public text', summary='A summary.',
        )

        assert await repos.context.has_summary(ctx_id, scope=LOCAL_SCOPE) is None

    @pytest.mark.asyncio
    async def test_reads_the_transaction_s_own_write(self, repos: RepositoryContainer) -> None:
        """Inside a transaction the read sees the summary the same transaction wrote."""
        async with repos.context.backend.begin_transaction(access_scope=LOCAL_SCOPE) as txn:
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE, visibility='private', thread_id='summary_thread', source='user',
                content_type='text', text_content='Written in the transaction', summary='Fresh.', txn=txn,
            )
            assert await repos.context.has_summary(ctx_id, scope=LOCAL_SCOPE, txn=txn) is True

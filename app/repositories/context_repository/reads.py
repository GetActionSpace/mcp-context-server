"""By-id reads, existence probes and id-prefix resolution for context entries."""

import operator
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import chunk_ids
from app.repositories.context_repository.records import EntryProbe
from app.repositories.context_repository.records import IdAccess
from app.repositories.context_repository.records import entry_columns

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

# The probe of an entry that does not exist for the caller, absent or unreadable alike.
_MISSING_PROBE = EntryProbe(False, None, None, None, False)


class ContextReadMixin(BaseRepository):
    """By-id reads and existence probes over ``context_entries``.

    Fetches entries by id in bounded chunks, probes an entry's existence, source,
    version, owner and write access for the update paths, probes the write and
    owner access of many entries at once, confirms and reads the content type of
    an entry the caller may modify, and resolves an id prefix to the matching
    full ids.
    """

    async def get_by_ids(self, context_ids: list[str], *, scope: Scope) -> list[Any]:
        """Get the context entries the scope may read among the given IDs.

        An id of an entry the scope may not read is skipped exactly like an id
        no entry carries, so the result never reveals that a hidden entry exists.

        Args:
            context_ids: List of context entry IDs
            scope: The caller's scope; only entries it may read are returned.

        Returns:
            List of context entry rows (sqlite3.Row or asyncpg.Record depending on
            backend), ordered by ``created_at DESC, id DESC`` across the whole result.
            Each row carries the entry columns, ``visibility`` among them, and
            ``is_owner``: 1 when the scope owns the entry, else 0.
        """
        # Defensive check: return empty list if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return []

        # Fetch in bounded chunks (mirroring delete_by_ids) so an arbitrarily long id
        # list never exceeds a backend's per-statement bound-parameter limit. When more
        # than one statement ran, the accumulated rows are re-sorted in Python so the
        # single-statement ORDER BY created_at DESC, id DESC contract holds across chunk
        # boundaries (the Python tuple sort compares the same TEXT/TIMESTAMPTZ created_at
        # and unique lowercase-hex id values the SQL ORDER BY compares).
        chunks = chunk_ids(context_ids)
        backend_type = self.backend.backend_type
        entries = self._readable('context_entries', alias='context_entries')

        def _chunk_statement(chunk: list[str]) -> tuple[str, list[object]]:
            # Placeholder order follows the text: the OWNER bind of the select list,
            # then the chunk's ids, then the READ binds of the WHERE clause.
            owner = build_access_predicate(
                scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='context_entries', start=1,
            )
            first_id = 1 + owner.bind_count
            placeholders = ','.join([self._placeholder(first_id + i) for i in range(len(chunk))])
            read = build_access_predicate(
                scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries',
                start=first_id + len(chunk),
            )
            # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
            query = f'''
                SELECT {entry_columns()}, {owner.admits_expression()} AS is_owner FROM {entries}
                WHERE id IN ({placeholders}){read.and_clause()}
                ORDER BY created_at DESC, id DESC
            '''
            return query, [*owner.params, *chunk, *read.params]

        if backend_type == 'sqlite':

            def _fetch_sqlite(conn: sqlite3.Connection) -> list[Any]:
                cursor = conn.cursor()
                rows: list[Any] = []
                for chunk in chunks:
                    query, params = _chunk_statement(chunk)
                    cursor.execute(query, tuple(params))
                    rows.extend(cursor.fetchall())
                if len(chunks) > 1:
                    rows.sort(key=operator.itemgetter('created_at', 'id'), reverse=True)
                return rows

            return await self.backend.execute_read(_fetch_sqlite, access_scope=scope)

        # PostgreSQL
        async def _fetch_postgresql(conn: 'asyncpg.Connection') -> list[Any]:
            rows: list[Any] = []
            for chunk in chunks:
                query, params = _chunk_statement(chunk)
                rows.extend(await conn.fetch(query, *params))
            if len(chunks) > 1:
                rows.sort(key=operator.itemgetter('created_at', 'id'), reverse=True)
            return rows

        return await self.backend.execute_read(_fetch_postgresql, access_scope=scope)

    async def check_entry_exists(self, context_id: str, *, scope: Scope) -> EntryProbe:
        """Check whether the scope may read a context entry and return its source, version, owner and write access.

        One statement reads the row under the READ predicate and computes the WRITE
        predicate in its select list, so an entry the scope may not read probes
        exactly like a missing one and never reveals its owner.

        Args:
            context_id: ID of the context entry
            scope: The caller's scope.

        Returns:
            An :class:`EntryProbe`. When ``exists`` is True, ``source`` is
            'user' or 'agent', ``version`` is the current optimistic-concurrency
            token -- update_context captures it BEFORE generation and passes it
            to update_context_entry as the compare-and-set guard, so a concurrent
            writer that commits during generation is detected -- ``owner_id``
            is the stamped owner backing the owner-only visibility-change check,
            and ``can_write`` says whether the scope may modify the entry's content.
        """
        backend_type = self.backend.backend_type
        # Placeholder order follows the text: the WRITE binds of the select list,
        # then the id, then the READ binds of the WHERE clause.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=1,
        )
        id_position = write.bind_count + 1
        read = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=id_position + 1,
        )
        entries = self._readable('context_entries', alias='context_entries')
        query = (
            f'SELECT source, version, owner_id, {write.admits_expression()} AS can_write FROM {entries} '
            f'WHERE id = {self._placeholder(id_position)}{read.and_clause()} LIMIT 1'
        )
        params: list[object] = [*write.params, context_id, *read.params]

        if backend_type == 'sqlite':

            def _check_exists_sqlite(conn: sqlite3.Connection) -> EntryProbe:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
                if row is None:
                    return _MISSING_PROBE
                return EntryProbe(
                    True,
                    cast(str, row['source']),
                    cast(int, row['version']),
                    cast(str, row['owner_id']),
                    bool(row['can_write']),
                )

            return await self.backend.execute_read(_check_exists_sqlite, access_scope=scope)

        # PostgreSQL
        async def _check_exists_postgresql(conn: 'asyncpg.Connection') -> EntryProbe:
            row = await conn.fetchrow(query, *params)
            if row is None:
                return _MISSING_PROBE
            return EntryProbe(
                True,
                cast(str, row['source']),
                cast(int, row['version']),
                cast(str, row['owner_id']),
                bool(row['can_write']),
            )

        return await self.backend.execute_read(_check_exists_postgresql, access_scope=scope)

    async def probe_ids(
        self,
        context_ids: list[str],
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> dict[str, IdAccess]:
        """Report what the scope may do with each entry it may read among the given IDs.

        Each statement reads one chunk of ids under the READ predicate and computes
        the WRITE and OWNER predicates in its select list. An id of an entry the
        scope may not read is absent from the result exactly like an id no entry
        carries, so a caller can tell a readable entry it may not modify or delete
        from one it may not see at all.

        Args:
            context_ids: IDs of the entries to probe.
            scope: The caller's scope.
            txn: Optional transaction context. When provided the probe runs on the
                transaction's own connection, so it sees the same rows the
                transaction then modifies.

        Returns:
            The access of every readable requested entry, keyed by its id.
        """
        if not context_ids:
            return {}

        backend_type = txn.backend_type if txn else self.backend.backend_type
        chunks = chunk_ids(context_ids)
        entries = self._readable('context_entries', alias='context_entries')

        def _chunk_statement(chunk: list[str]) -> tuple[str, list[object]]:
            # Placeholder order follows the text: the WRITE and OWNER binds of the
            # select list, then the chunk's ids, then the READ binds of the WHERE clause.
            write = build_access_predicate(
                scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=1,
            )
            owner = build_access_predicate(
                scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='context_entries',
                start=1 + write.bind_count,
            )
            first_id = 1 + write.bind_count + owner.bind_count
            placeholders = ','.join([self._placeholder(first_id + i) for i in range(len(chunk))])
            read = build_access_predicate(
                scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries',
                start=first_id + len(chunk),
            )
            query = (
                f'SELECT id, {write.admits_expression()} AS can_write, {owner.admits_expression()} AS is_owner '
                f'FROM {entries} WHERE id IN ({placeholders}){read.and_clause()}'
            )
            return query, [*write.params, *owner.params, *chunk, *read.params]

        if backend_type == 'sqlite':

            def _probe_sqlite(conn: sqlite3.Connection) -> dict[str, IdAccess]:
                cursor = conn.cursor()
                access: dict[str, IdAccess] = {}
                for chunk in chunks:
                    query, params = _chunk_statement(chunk)
                    cursor.execute(query, tuple(params))
                    for row in cursor.fetchall():
                        access[str(row['id'])] = IdAccess(bool(row['can_write']), bool(row['is_owner']))
                return access

            if txn is not None:
                return await self._run_sqlite_txn(_probe_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_probe_sqlite, access_scope=scope)

        # PostgreSQL
        async def _probe_postgresql(conn: 'asyncpg.Connection') -> dict[str, IdAccess]:
            access: dict[str, IdAccess] = {}
            for chunk in chunks:
                query, params = _chunk_statement(chunk)
                for row in await conn.fetch(query, *params):
                    # normalize_id(str(...)) keys the result by the canonical hex id the
                    # caller passed, also on a connection without the uuid codec.
                    access[normalize_id(str(row['id']))] = IdAccess(bool(row['can_write']), bool(row['is_owner']))
            return access

        if txn is not None:
            return await _probe_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_probe_postgresql, access_scope=scope)

    async def entry_exists(
        self,
        context_id: str,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> bool:
        """Return whether the scope may modify a context entry, optionally on a transaction connection.

        The write gate of the update paths: a lightweight companion to
        check_entry_exists for callers that need only presence under the WRITE
        predicate and must run inside an open transaction.
        execute_update_in_transaction uses it to confirm the parent row before a
        tags-only or images-only update, whose child writes would otherwise
        violate the foreign key against a missing parent (charging the circuit
        breaker), orphan the rows, or modify an entry the caller may not modify;
        the atomic batch update uses it to tell a lost write from a version
        conflict. An entry the scope may not modify returns False exactly like a
        missing one.

        When invoked inside a transaction on PostgreSQL, the presence check locks
        the parent row with FOR KEY SHARE OF context_entries so a concurrent DELETE
        blocks until this update transaction commits: without the lock the row can
        be deleted in the window between this check and the subsequent child
        tag/image writes, whose foreign key then fails -- a non-ControlFlowError
        that charges the circuit breaker. FOR KEY SHARE permits concurrent non-key
        updates while blocking row deletion, which is exactly the parent-presence
        guarantee the child writes need; naming context_entries confines the lock
        to that parent row. Outside a transaction the lock
        would release at statement end and serve no purpose, so it is applied only
        on the transaction path.

        Args:
            context_id: ID of the context entry.
            scope: The caller's scope; only an entry it may modify counts as present.
            txn: Optional transaction context. When provided the read runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            True if the entry exists and the scope may modify it, False otherwise.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        # The id binds first, the WRITE predicate next.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=2,
        )
        lock_clause = ' FOR KEY SHARE OF context_entries' if txn is not None and backend_type == 'postgresql' else ''
        query = (
            f'SELECT 1 FROM context_entries WHERE id = {self._placeholder(1)}{write.and_clause()} LIMIT 1{lock_clause}'
        )
        params: list[object] = [context_id, *write.params]

        if backend_type == 'sqlite':

            def _entry_exists_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                return cursor.fetchone() is not None

            if txn is not None:
                return await self._run_sqlite_txn(_entry_exists_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_entry_exists_sqlite, access_scope=scope)

        # PostgreSQL
        async def _entry_exists_postgresql(conn: 'asyncpg.Connection') -> bool:
            row = await conn.fetchrow(query, *params)
            return row is not None

        if txn is not None:
            return await _entry_exists_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_entry_exists_postgresql, access_scope=scope)

    async def get_content_type(
        self,
        context_id: str,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> str | None:
        """Get the content type of a context entry the scope may modify.

        Args:
            context_id: ID of the context entry
            scope: The caller's scope; an entry it may not modify reads as missing.
            txn: Optional transaction context. When provided the read runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            Content type ('text' or 'multimodal'), or None if the entry does not
            exist or the scope may not modify it
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        # The id binds first, the WRITE predicate next.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=2,
        )
        entries = self._readable('context_entries', alias='context_entries')
        query = f'SELECT content_type FROM {entries} WHERE id = {self._placeholder(1)}{write.and_clause()}'
        params: list[object] = [context_id, *write.params]

        if backend_type == 'sqlite':

            def _get_content_type_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
                return row['content_type'] if row else None

            if txn is not None:
                return await self._run_sqlite_txn(_get_content_type_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_get_content_type_sqlite, access_scope=scope)

        # PostgreSQL
        async def _get_content_type_postgresql(conn: 'asyncpg.Connection') -> str | None:
            row = await conn.fetchrow(query, *params)
            return row['content_type'] if row else None

        if txn is not None:
            return await _get_content_type_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_get_content_type_postgresql, access_scope=scope)

    async def has_summary(
        self,
        context_id: str,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> bool | None:
        """Report whether an entry the scope may modify holds a summary.

        Args:
            context_id: ID of the context entry.
            scope: The caller's scope; an entry it may not modify reads as missing.
            txn: Optional transaction context; the read then sees the transaction's own
                writes, on its connection.

        Returns:
            True when the entry holds a summary, False when it holds none, or None if the
            entry does not exist or the scope may not modify it.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        # The id binds first, the WRITE predicate next.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=2,
        )
        entries = self._readable('context_entries', alias='context_entries')
        query = (
            f'SELECT summary IS NOT NULL AS has_summary FROM {entries} '
            f'WHERE id = {self._placeholder(1)}{write.and_clause()}'
        )
        params: list[object] = [context_id, *write.params]

        if backend_type == 'sqlite':

            def _has_summary_sqlite(conn: sqlite3.Connection) -> bool | None:
                row = conn.execute(query, tuple(params)).fetchone()
                return None if row is None else bool(row['has_summary'])

            if txn is not None:
                return await self._run_sqlite_txn(_has_summary_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_has_summary_sqlite, access_scope=scope)

        async def _has_summary_postgresql(conn: 'asyncpg.Connection') -> bool | None:
            row = await conn.fetchrow(query, *params)
            return None if row is None else bool(row['has_summary'])

        if txn is not None:
            return await _has_summary_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_has_summary_postgresql, access_scope=scope)

    async def find_ids_by_prefix(self, prefix: str, limit: int = 2, *, scope: Scope) -> list[str]:
        """Find the IDs of entries the scope may read that begin with the given prefix.

        Backs the ID-prefix resolution helper :func:`app.ids.resolve_prefix` (reached via
        :func:`app.ids.resolve_or_normalize_id`), which expands a short user-supplied
        prefix into a full id when ambiguity is unlikely. The caller decides what to do
        when ``limit`` rows are returned; this method's contract is "return up to N
        matches". The access predicate applies before ``LIMIT``, so an entry the scope
        may not read never takes a slot from a readable match and never makes a prefix
        ambiguous.

        Args:
            prefix: Lowercase hex prefix. ``resolve_prefix`` only calls this for prefixes
                of 8-31 hex characters (the range :func:`app.ids.is_id_prefix` accepts);
                the caller is responsible for normalization and that length check.
            limit: Maximum number of IDs to return. Defaults to 2 so callers can
                detect ambiguity by checking ``len(result) > 1``.
            scope: The caller's scope; only entries it may read match.

        Returns:
            List of matching context_id strings (UUIDv7 hex, 32 chars), up to ``limit``.
        """
        backend_type = self.backend.backend_type
        # The prefix binds first, the access predicate next, LIMIT last.
        read = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=2,
        )
        limit_placeholder = self._placeholder(2 + read.bind_count)
        params: list[object] = [prefix + '%', *read.params, limit]
        entries = self._readable('context_entries', alias='context_entries')

        if backend_type == 'sqlite':

            def _find_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT id FROM {entries}
                    WHERE id LIKE {self._placeholder(1)}{read.and_clause()}
                    ORDER BY id
                    LIMIT {limit_placeholder}
                    ''',
                    tuple(params),
                )
                return [row['id'] for row in cursor.fetchall()]

            return await self.backend.execute_read(_find_sqlite, access_scope=scope)

        # PostgreSQL: id is a uuid column; canonical text form contains hyphens
        # which are absent from the user-supplied hex prefix. REPLACE() removes
        # hyphens so LIKE matches against the 32-char hex representation.
        async def _find_postgresql(conn: 'asyncpg.Connection') -> list[str]:
            rows = await conn.fetch(
                f'''
                SELECT id FROM {entries}
                WHERE REPLACE(CAST(id AS TEXT), '-', '') LIKE {self._placeholder(1)}{read.and_clause()}
                ORDER BY id
                LIMIT {limit_placeholder}
                ''',
                *params,
            )
            # The pool's uuid->str codec (registered in init_pool_connection) already
            # decodes id to the 32-char hex the SQLite path returns.
            # normalize_id(str(...)) is idempotent defense-in-depth for
            # codec-less connections, so prefix resolution echoes a canonical
            # context_id on both backends (mirrors grep_scan_text_contents).
            return [normalize_id(str(row['id'])) for row in rows]

        return await self.backend.execute_read(_find_postgresql, access_scope=scope)

"""The reads and writes of context entries that generating their outputs needs.

The generation of an entry's outputs reads the entry's text and which outputs it already
holds, and writes a generated summary only while the entry still holds the text it read.
Both statements run under the identity the caller passes and carry no access predicate:
outputs are generated for whichever entry holds a generation job, and the caller that
enqueued the job was authorized to write the entry.
"""

import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import cast

from app.backends.base import BackendAccess
from app.repositories.base import BaseRepository
from app.repositories.context_repository.records import GenerationSource

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

_EMBEDDINGS_TABLE = 'embedding_metadata'
_NODES_TABLE = 'context_index_nodes'


class _Row(Protocol):
    """A row read by column name: a ``sqlite3.Row`` or an ``asyncpg.Record``."""

    def __getitem__(self, key: str, /) -> object: ...


class ContextGenerationMixin(BaseRepository):
    """Generation reads and writes over ``context_entries``.

    ``read_generation_source`` reads what generating an entry's outputs starts from;
    ``write_generation_summary`` stores a generated summary, guarded on the text it was
    generated from.
    """

    def _generation_source_sql(self, *, embeddings: bool, nodes: bool) -> str:
        """Render the read of an entry's generation source.

        Args:
            embeddings: Whether the embedding table exists, so the read reports stored embeddings.
            nodes: Whether the index-tree node table exists, so the read counts stored nodes.

        Returns:
            The statement, binding the entry id.
        """
        columns = ['ce.text_content', 'ce.source', 'ce.content_hash', 'ce.summary IS NOT NULL AS has_summary']
        if embeddings:
            columns.append(f'EXISTS (SELECT 1 FROM {_EMBEDDINGS_TABLE} em WHERE em.context_id = ce.id) AS has_embeddings')
        if nodes:
            columns.append(f'(SELECT COUNT(*) FROM {_NODES_TABLE} n WHERE n.context_id = ce.id) AS node_count')
        return f'SELECT {", ".join(columns)} FROM context_entries ce WHERE ce.id = {self._placeholder(1)}'

    async def read_generation_source(self, context_id: str, *, scope: BackendAccess) -> GenerationSource | None:
        """Read an entry's text and which generated outputs it already holds.

        The embedding and node tables exist only when their features were provisioned;
        an entry reports no embeddings or no nodes when its table is absent.

        Args:
            context_id: The entry's id.
            scope: The identity the read runs as.

        Returns:
            The entry's generation source, or None when no entry carries the id.
        """

        def _source(row: _Row, *, embeddings: bool, nodes: bool) -> GenerationSource:
            return GenerationSource(
                text_content=str(row['text_content'] or ''),
                source=str(row['source']),
                content_hash=None if row['content_hash'] is None else str(row['content_hash']),
                has_summary=bool(row['has_summary']),
                has_embeddings=bool(row['has_embeddings']) if embeddings else False,
                node_count=int(str(row['node_count'])) if nodes else 0,
            )

        if self.backend.backend_type == 'sqlite':

            def _read_sqlite(conn: sqlite3.Connection) -> GenerationSource | None:
                present = {
                    str(row['name'])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
                        (_EMBEDDINGS_TABLE, _NODES_TABLE),
                    ).fetchall()
                }
                embeddings, nodes = _EMBEDDINGS_TABLE in present, _NODES_TABLE in present
                sql = self._generation_source_sql(embeddings=embeddings, nodes=nodes)
                row = conn.execute(sql, (context_id,)).fetchone()
                return None if row is None else _source(row, embeddings=embeddings, nodes=nodes)

            return await self.backend.execute_read(_read_sqlite, access_scope=scope)

        async def _read_postgresql(conn: 'asyncpg.Connection') -> GenerationSource | None:
            # to_regclass resolves each name through the connection's search_path, as the
            # bare names of the read do, and is NULL for a missing table.
            tables = await conn.fetchrow(
                'SELECT to_regclass($1) IS NOT NULL AS embeddings, to_regclass($2) IS NOT NULL AS nodes',
                _EMBEDDINGS_TABLE,
                _NODES_TABLE,
            )
            embeddings = bool(tables is not None and tables['embeddings'])
            nodes = bool(tables is not None and tables['nodes'])
            sql = self._generation_source_sql(embeddings=embeddings, nodes=nodes)
            row = await conn.fetchrow(sql, context_id)
            return None if row is None else _source(row, embeddings=embeddings, nodes=nodes)

        return await self.backend.execute_read(cast(Any, _read_postgresql), access_scope=scope)

    async def write_generation_summary(
        self,
        context_id: str,
        *,
        content_hash: str | None,
        summary: str | None,
        txn: 'TransactionContext',
    ) -> bool:
        """Store a generated summary while the entry still holds the text it was generated from.

        The statement writes the summary, keeping the stored one when ``summary`` is None,
        and advances ``updated_at``, both only while the entry's ``content_hash`` is still
        ``content_hash``: as the first write of a completion it also takes the entry's row
        lock before the completion writes anything else. It writes neither the text, the
        version nor the access columns, so the full-text index, the compare-and-set token
        and the access guard are left as they are.

        Args:
            context_id: The entry's id.
            content_hash: The hash of the text the outputs were generated from, as the
                generation source reported it.
            summary: The generated summary, or None when none was generated.
            txn: The transaction that writes the generated outputs.

        Returns:
            Whether the entry still holds that text; False when its text changed or it is gone.
        """
        placeholder = self._placeholder
        same_hash = 'IS' if txn.backend_type == 'sqlite' else 'IS NOT DISTINCT FROM'
        sql = (
            f'UPDATE context_entries SET summary = COALESCE({placeholder(1)}, summary), updated_at = CURRENT_TIMESTAMP '
            f'WHERE id = {placeholder(2)} AND content_hash {same_hash} {placeholder(3)}'
        )
        params = [summary, context_id, content_hash]

        if txn.backend_type == 'sqlite':

            def _write_sqlite(conn: sqlite3.Connection) -> bool:
                return conn.execute(sql, params).rowcount > 0

            return await self._run_sqlite_txn(_write_sqlite, cast(sqlite3.Connection, txn.connection))

        result = await cast('asyncpg.Connection', txn.connection).execute(sql, *params)
        return int(result.split()[-1]) > 0 if result else False

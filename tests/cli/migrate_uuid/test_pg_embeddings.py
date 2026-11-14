"""Tests for the PostgreSQL-to-PostgreSQL copies of embedding metadata and fp32 vectors."""

from collections.abc import Awaitable
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import cast

import pytest

from app.cli.migrate_uuid.pg_embeddings import copy_embedding_metadata_pg
from app.cli.migrate_uuid.pg_embeddings import copy_vec_embeddings_pg
from app.cli.migrate_uuid.records import MigrationStats

NEW_ID_1 = '0190abcdef1234567890abcdef000001'
NEW_ID_2 = '0190abcdef1234567890abcdef000002'

type EmbeddingCopy = Callable[..., Awaitable[None]]


def _source_row(context_id: int) -> dict[str, object]:
    """Return a source row carrying every column either copy reads."""
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    return {
        'context_id': context_id,
        'model_name': 'fake-model',
        'dimensions': 3,
        'created_at': stamp,
        'updated_at': stamp,
        'chunk_count': 1,
        'embedding': '[0.1,0.2,0.3]',
        'start_index': 0,
        'end_index': 7,
    }


class _SourceConn:
    """Source connection whose column probes find every column and whose fetch returns the given rows."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    async def fetchval(self, _sql: str, *_args: object) -> bool:
        return True

    async def fetch(self, _sql: str, *_args: object) -> list[dict[str, object]]:
        return self._rows


class _TargetConn:
    """Target connection whose probes find every table and column, recording the bound values of each insert."""

    def __init__(self) -> None:
        self.inserted: list[tuple[object, ...]] = []

    async def fetchval(self, _sql: str, *_args: object) -> bool:
        return True

    async def execute(self, _sql: str, *args: object) -> str:
        self.inserted.append(args)
        return 'INSERT 0 1'


@pytest.mark.asyncio
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize(
    ('copy', 'table', 'counter'),
    [
        (copy_embedding_metadata_pg, 'embedding_metadata', 'embedding_metadata_migrated'),
        (copy_vec_embeddings_pg, 'vec_context_embeddings', 'vec_rows_migrated'),
    ],
)
async def test_skipped_parent_and_missing_parent_rows_are_left_out_with_their_own_warnings(
    copy: EmbeddingCopy, table: str, counter: str, dry_run: bool,
) -> None:
    """A row of a skipped entry names the skip, a row of an absent entry names the missing id, and neither is copied."""
    target = _TargetConn()
    stats = MigrationStats()

    await copy(
        cast(Any, _SourceConn([_source_row(1), _source_row(2), _source_row(3)])),
        cast(Any, target),
        {1: NEW_ID_1, 2: NEW_ID_2},
        stats,
        dry_run,
        skipped_ids=frozenset({2}),
        probe_target=True,
    )

    assert stats.warnings == [
        f'{table} row context_id=2 skipped: parent context_entries row was skipped',
        f'{table} row references missing context_id=3; skipped',
    ]
    assert getattr(stats, counter) == 1
    assert [args[0] for args in target.inserted] == ([] if dry_run else [NEW_ID_1])

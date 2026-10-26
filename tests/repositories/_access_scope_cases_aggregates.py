"""Two-principal cases of the aggregate and discovery seams (A1-A6).

Every aggregate applies the caller's READ predicate in SQL before GROUP BY, ORDER BY and
LIMIT, so each figure is computed over the rows the caller may read, and a top-N list holds
the caller's own top N rather than whatever readable rows survive a global top N:

- A1: ``StatisticsRepository.get_thread_list`` lists only the threads holding a readable row,
  with the entry count, source count, multimodal count, first and last creation time and
  last id computed over readable rows, ordered by the latest readable activity. A hidden row
  that is the newest of its thread moves that thread up for its readers only. Variants: the
  full listing; and page-fill, where more hidden threads than one page hold newer activity
  than every readable thread, yet each page is full of readable threads in order.
- A2: ``get_database_statistics`` reports the totals, breakdowns, the five most active
  threads and the ten most used tags over readable rows. Variants: the seed; top-N fill,
  where more than five hidden threads and more than ten hidden tags carry higher counts than
  the readable ones, yet both lists hold the caller's own top items; and excepted keys, where
  ``database_size_mb`` stays the deployment-level figure every caller sees alike.
- A3: ``get_summary_statistics`` counts readable rows and the readable rows with a summary.
- A4: ``EmbeddingRepository.get_statistics`` counts readable rows, the readable rows with an
  embedding and their chunks.
- A5: ``FtsRepository.get_statistics`` counts readable rows and readable indexed rows; the
  system scope, which sizes the full-text migration estimate, counts every row.
- A6: ``IndexNodeRepository.count_all_nodes`` counts the index-tree nodes of readable rows.

Expectations come from a declared row table: every row names the scopes that read it, and a
small oracle derives each figure from the rows a scope reads.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Any

import numpy as np

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import Scope
from app.backends.base import MAINTENANCE
from app.backends.sqlite_backend import SQLiteBackend
from app.compression.factory import get_cached_compression_provider
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import SEED_THREAD
from tests.repositories._access_scope_cases import TEAM_GROUP
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Setup
from tests.repositories._access_scope_cases import Source
from tests.repositories._access_scope_cases import Visibility
from tests.repositories._access_scope_cases import per_scope
from tests.repositories._access_scope_cases import seed_text
from tests.repositories._access_scope_cases import vector_at

type ThreadObservable = tuple[str, int, int, int, str, str, str]
type ThreadPages = tuple[tuple[str, ...], tuple[str, ...]]

# The readers of a row: everyone for a public row, its owner and the system scope for a
# private one, plus carol for a private alice row read-granted to her group.
_EVERYONE: tuple[str, ...] = tuple(SCOPES)
_ALICE_ONLY = ('alice', 'system')
_BOB_ONLY = ('bob', 'system')
_ALICE_AND_TEAM = ('alice', 'carol', 'system')

THREAD_PAGE_SIZE = 2
TOP_THREADS = 5
TOP_TAGS = 10


@dataclass(frozen=True, slots=True)
class _Row:
    """One entry an aggregate counts: its owner, where it sits, what it carries and who reads it.

    Attributes:
        label: Unique label of the entry.
        owner: Principal stamped as the owner.
        visibility: The entry's visibility.
        readers: The scope names that read the entry.
        thread_id: Thread of the entry.
        source: Source of the entry.
        image: Whether the entry carries one image.
        tags: The entry's tags; empty stores the default ``seed`` and ``owner-<owner>``.
        grants: Grants on the entry.
        second: Creation time in seconds after midnight of the reference day, for the
            cases that pin creation times.
    """

    label: str
    owner: str
    visibility: Visibility
    readers: tuple[str, ...]
    thread_id: str = SEED_THREAD
    source: Source = SEED_SOURCE
    image: bool = False
    tags: tuple[str, ...] = ()
    grants: tuple[Grant, ...] = ()
    second: int = 0

    @property
    def stored_tags(self) -> tuple[str, ...]:
        """The tags the entry is stored with."""
        return self.tags or ('seed', f'owner-{self.owner}')


# The seed rows as the oracle sees them, created one second apart in seed order.
_SEED_MODEL: tuple[_Row, ...] = tuple(
    _Row(
        row.label, row.owner, row.visibility,
        readers=tuple(name for name in SCOPES if row.label in READABLE[name]),
        image=row.image, grants=row.grants, second=position,
    )
    for position, row in enumerate(SEED_ROWS)
)

# A1: one thread per kind of reader, a thread mixing a public user row with a newer hidden
# agent row, and a hidden row that is the newest of the seed thread.
_A1_ROWS: tuple[_Row, ...] = (
    _Row('a1_hidden', 'alice', 'private', _ALICE_ONLY, thread_id='a1-hidden', second=20),
    _Row('a1_bob', 'bob', 'private', _BOB_ONLY, thread_id='a1-bob', image=True, second=21),
    _Row(
        'a1_team', 'alice', 'private', _ALICE_AND_TEAM, thread_id='a1-team',
        grants=(Grant('group', TEAM_GROUP, 'read'),), second=22,
    ),
    _Row('a1_mixed_public', 'alice', 'public', _EVERYONE, thread_id='a1-mixed', source='user', second=23),
    _Row('a1_mixed_hidden', 'alice', 'private', _ALICE_ONLY, thread_id='a1-mixed', image=True, second=24),
    _Row('a1_seed_hidden', 'alice', 'private', _ALICE_ONLY, second=30),
)

# A1 page-fill: hidden threads whose activity is newer than every readable thread.
_A1_FILL_ROWS: tuple[_Row, ...] = (
    *_A1_ROWS,
    *(
        _Row(f'a1_fill_{index}', 'alice', 'private', _ALICE_ONLY, thread_id=f'a1-fill-{index}', second=40 + index)
        for index in range(5)
    ),
)

# A2 top-N fill: five public single-row threads with two tags each (one row from the user),
# and six hidden two-row threads whose rows carry eleven hidden tags, so every hidden thread
# and tag out-counts every public one.
_HIDDEN_TAGS: tuple[str, ...] = tuple(f'h-tag-{index:02d}' for index in range(11))
_A2_ROWS: tuple[_Row, ...] = (
    *(
        _Row(
            f'a2_visible_{index}', 'alice', 'public', _EVERYONE, thread_id=f'a2-visible-{index}',
            source='user' if index == 0 else SEED_SOURCE, tags=(f'v-tag-{2 * index}', f'v-tag-{2 * index + 1}'),
        )
        for index in range(5)
    ),
    *(
        _Row(
            f'a2_hidden_{thread}_{row}', 'alice', 'private', _ALICE_ONLY, thread_id=f'a2-hidden-{thread}',
            image=thread == 0 and row == 0, tags=_HIDDEN_TAGS,
        )
        for thread in range(6)
        for row in range(2)
    ),
)


def _stamp(second: int) -> str:
    """The creation time a listing reports for a row created ``second`` seconds after the reference midnight."""
    return f'2026-01-01T00:00:{second:02d}Z'


async def _set_created_at(db: ScopedDb, label: str, second: int) -> None:
    """Set the creation time of one row to ``second`` seconds after the reference midnight."""
    stamp = f'2026-01-01 00:00:{second:02d}'
    if db.backend.backend_type == 'sqlite':
        await db.execute('UPDATE context_entries SET created_at = ? WHERE id = ?', (stamp, db.ids[label]))
        return
    await db.execute(
        'UPDATE context_entries SET created_at = CAST(CAST(? AS TEXT) AS TIMESTAMPTZ) WHERE id = ?',
        (f'{stamp}+00', db.ids[label]),
    )


def _add_rows(rows: Sequence[_Row], *, pin_creation_times: bool = False) -> Setup:
    """Return a setup storing ``rows`` in order and, when asked, pinning every creation time.

    Args:
        rows: The rows to add after the seed.
        pin_creation_times: Set the creation time of every seed and added row to its ``second``.

    Returns:
        The setup hook.
    """

    async def _setup(db: ScopedDb) -> None:
        for row in rows:
            await db.add_entry(
                row.label, owner=row.owner, visibility=row.visibility, thread_id=row.thread_id,
                source=row.source, tags=row.stored_tags, image=row.image, grants=row.grants,
            )
        if pin_creation_times:
            for row in (*_SEED_MODEL, *rows):
                await _set_created_at(db, row.label, row.second)

    return _setup


def _readable(rows: Sequence[_Row], scope_name: str) -> list[_Row]:
    """The rows the scope reads, in insertion order."""
    return [row for row in rows if scope_name in row.readers]


def _expected_threads(rows: Sequence[_Row], scope_name: str) -> tuple[ThreadObservable, ...]:
    """The threads holding a readable row, with figures over readable rows, latest activity first.

    Rows are inserted in order with rising creation times, so the newest readable row of a
    thread is both its latest activity and its largest id.

    Args:
        rows: The seed and added rows, in insertion order.
        scope_name: The scope, a key of :data:`SCOPES`.

    Returns:
        One listing entry per thread, with the last id as a label.
    """
    members: dict[str, list[_Row]] = {}
    for row in _readable(rows, scope_name):
        members.setdefault(row.thread_id, []).append(row)
    listing = [
        (
            thread_id, len(thread_rows), len({row.source for row in thread_rows}),
            sum(row.image for row in thread_rows), _stamp(thread_rows[0].second), _stamp(thread_rows[-1].second),
            thread_rows[-1].label,
        )
        for thread_id, thread_rows in members.items()
    ]
    return tuple(sorted(listing, key=itemgetter(5), reverse=True))


def _expected_pages(scope_name: str) -> ThreadPages:
    """The first two pages of the readable threads of the page-fill rows."""
    order = tuple(thread[0] for thread in _expected_threads((*_SEED_MODEL, *_A1_FILL_ROWS), scope_name))
    return order[:THREAD_PAGE_SIZE], order[THREAD_PAGE_SIZE:2 * THREAD_PAGE_SIZE]


def _top(counts: Counter[str], size: int) -> list[tuple[str, int]]:
    """The ``size`` largest counts, ties broken by key in byte order."""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:size]


def _expected_database_figures(rows: Sequence[_Row], scope_name: str) -> dict[str, Any]:
    """The scoped figures of ``get_database_statistics`` over the rows the scope reads."""
    readable = _readable(rows, scope_name)
    thread_counts = Counter(row.thread_id for row in readable)
    tag_counts = Counter(tag for row in readable for tag in row.stored_tags)
    return {
        'total_entries': len(readable),
        'by_source': dict(Counter(row.source for row in readable)),
        'by_content_type': dict(Counter('multimodal' if row.image else 'text' for row in readable)),
        'total_images': sum(row.image for row in readable),
        'unique_tags': len(tag_counts),
        'total_threads': len(thread_counts),
        'avg_entries_per_thread': round(len(readable) / len(thread_counts), 2),
        'most_active_threads': [
            {'thread_id': thread_id, 'count': count} for thread_id, count in _top(thread_counts, TOP_THREADS)
        ],
        'top_tags': [{'tag': tag, 'count': count} for tag, count in _top(tag_counts, TOP_TAGS)],
    }


async def _thread_listing(db: ScopedDb, scope: Scope) -> tuple[ThreadObservable, ...]:
    """List every thread as the scope, with the last id as a label."""
    threads = await db.repos.statistics.get_thread_list(scope=scope)
    return tuple(
        (
            thread['thread_id'], int(thread['entry_count']), int(thread['source_types']),
            int(thread['multimodal_count']), thread['first_entry'], thread['last_entry'],
            db.labels_of([str(thread['last_id'])])[0],
        )
        for thread in threads
    )


async def _thread_pages(db: ScopedDb, scope: Scope) -> ThreadPages:
    """List the first two pages of :data:`THREAD_PAGE_SIZE` threads as the scope."""
    statistics = db.repos.statistics
    first = await statistics.get_thread_list(scope=scope, limit=THREAD_PAGE_SIZE, offset=0)
    second = await statistics.get_thread_list(scope=scope, limit=THREAD_PAGE_SIZE, offset=THREAD_PAGE_SIZE)
    return tuple(thread['thread_id'] for thread in first), tuple(thread['thread_id'] for thread in second)


async def _database_figures(db: ScopedDb, scope: Scope) -> dict[str, Any]:
    """Return the scoped figures of ``get_database_statistics`` as the scope."""
    stats = await db.repos.statistics.get_database_statistics(scope=scope)
    return {key: stats[key] for key in _expected_database_figures(_SEED_MODEL, 'system')}


async def _database_size_is_deployment_wide(db: ScopedDb, scope: Scope) -> tuple[int, bool]:
    """Return the scope's entry total and whether its database size is the system scope's.

    The size is read for the system scope right before and right after the scope's call, so
    a size that moves between the reads, in either direction, still brackets the scope's reading.

    Args:
        db: The seeded database.
        scope: The scope to read as.

    Returns:
        The scope's ``total_entries`` and whether its ``database_size_mb`` lies between the
        system scope's two readings.
    """
    db_path = db.backend.db_path if isinstance(db.backend, SQLiteBackend) else None
    statistics = db.repos.statistics
    before = await statistics.get_database_statistics(scope=SYSTEM_SCOPE, db_path=db_path)
    stats = await statistics.get_database_statistics(scope=scope, db_path=db_path)
    after = await statistics.get_database_statistics(scope=SYSTEM_SCOPE, db_path=db_path)
    sizes = (before['database_size_mb'], after['database_size_mb'])
    return stats['total_entries'], min(sizes) <= stats['database_size_mb'] <= max(sizes)


# A3: summaries on rows of every kind of reader, an empty summary that does not count, and a
# hidden summarized row.
_SUMMARIES = {
    'alice_private': 'Summary of alice_private.',
    'alice_public': 'Summary of alice_public.',
    'bob_private': 'Summary of bob_private.',
    'alice_public_dave_read': '',
}
_A3_HIDDEN = 'a3_hidden_summarized'


async def _add_summaries(db: ScopedDb) -> None:
    """Store the seed summaries and a hidden summarized row."""
    for label, summary in _SUMMARIES.items():
        await db.execute('UPDATE context_entries SET summary = ? WHERE id = ?', (summary, db.ids[label]))
    await db.add_entry(_A3_HIDDEN, owner='alice', visibility='private', summary=f'Summary of {_A3_HIDDEN}.')


def _expected_summary_figures(scope_name: str) -> dict[str, Any]:
    """Summary count, entry total and coverage over the rows the scope reads."""
    readable = (*READABLE[scope_name], *((_A3_HIDDEN,) if scope_name in _ALICE_ONLY else ()))
    summarized = sum(1 for label in readable if _SUMMARIES.get(label, label == _A3_HIDDEN))
    return {
        'summary_count': summarized,
        'total_entries': len(readable),
        'coverage_percentage': round(summarized / len(readable) * 100, 2),
    }


async def _summary_figures(db: ScopedDb, scope: Scope) -> dict[str, Any]:
    """Return ``get_summary_statistics`` as the scope."""
    return await db.repos.statistics.get_summary_statistics(scope=scope)


# A4: a hidden row with several chunks and a public row without an embedding.
_A4_CHUNKS = 3


async def _store_chunks(db: ScopedDb, context_id: str, label: str, count: int) -> None:
    """Store ``count`` chunk embeddings for one entry through the seed channel, compressed on the compressed layout."""
    text = seed_text(label)
    vectors = [vector_at(0.1 * index) for index in range(count)]
    payloads: list[bytes | None] = [None] * count
    if db.layout == 'compressed':
        provider = await get_cached_compression_provider()
        payloads = [provider.encode_sync(np.asarray([vector], dtype=np.float32)) for vector in vectors]
    chunks = [
        ChunkEmbedding(embedding=vector, start_index=0, end_index=len(text), payload=payload)
        for vector, payload in zip(vectors, payloads, strict=True)
    ]
    await db.seed.embeddings.store_chunked(context_id, chunks, model=get_settings().embedding.model, scope=MAINTENANCE)


async def _add_embedding_rows(db: ScopedDb) -> None:
    """Store a hidden row with several chunks and a public row without an embedding."""
    context_id = await db.add_entry('a4_hidden_chunks', owner='alice', visibility='private')
    await _store_chunks(db, context_id, 'a4_hidden_chunks', _A4_CHUNKS)
    await db.add_entry('a4_public_bare', owner='alice', visibility='public')


def _expected_embedding_figures(scope_name: str) -> dict[str, Any]:
    """Embedding figures over the rows the scope reads; every seed row has one chunk."""
    seed = len(READABLE[scope_name])
    hidden = int(scope_name in _ALICE_ONLY)
    embedded, chunks, entries = seed + hidden, seed + _A4_CHUNKS * hidden, seed + hidden + 1
    return {
        'total_embeddings': embedded,
        'total_entries': entries,
        'total_chunks': chunks,
        'average_chunks_per_entry': round(chunks / embedded, 2),
        'coverage_percentage': round(embedded / entries * 100, 2),
    }


async def _embedding_figures(db: ScopedDb, scope: Scope) -> dict[str, Any]:
    """Return ``EmbeddingRepository.get_statistics`` as the scope, without the backend name."""
    stats = await db.repos.embeddings.get_statistics(scope=scope)
    return {key: value for key, value in stats.items() if key != 'backend'}


async def _add_hidden_row(db: ScopedDb) -> None:
    """Store one hidden row, indexed for full-text search like every row."""
    await db.add_entry('a5_hidden', owner='alice', visibility='private')


def _expected_fts_figures(scope_name: str) -> dict[str, Any]:
    """Every readable row is indexed, so both counts are the readable total."""
    readable = len(READABLE[scope_name]) + int(scope_name in _ALICE_ONLY)
    return {'total_entries': readable, 'indexed_entries': readable, 'coverage_percentage': 100.0}


async def _fts_figures(db: ScopedDb, scope: Scope) -> dict[str, Any]:
    """Return ``FtsRepository.get_statistics`` as the scope, without the backend and engine names."""
    stats = await db.repos.fts.get_statistics(scope=scope)
    return {key: value for key, value in stats.items() if key not in ('backend', 'engine')}


# A6: a hidden row with several index-tree nodes and a public row without any.
_A6_NODES = 3


async def _add_node_rows(db: ScopedDb) -> None:
    """Store a hidden row with several index-tree nodes, written through the seed channel, and a public row without any."""
    label = 'a6_hidden_nodes'
    context_id = await db.add_entry(label, owner='alice', visibility='private')
    nodes = [
        IndexNodeRow(
            node_id=f'{label}-{index}', level=1, ordinal=index + 1, title=f'{label} {index}',
            node_summary=f'Summary of node {index}.', char_start=index, char_end=index + 1,
        )
        for index in range(_A6_NODES)
    ]
    await db.seed.index_nodes.replace_nodes_for_context(context_id, nodes, scope=MAINTENANCE)
    await db.add_entry('a6_public_bare', owner='alice', visibility='public')


async def _node_count(db: ScopedDb, scope: Scope) -> int:
    """Return ``count_all_nodes`` as the scope."""
    return await db.repos.index_nodes.count_all_nodes(scope=scope)


AGGREGATE_CASES: tuple[AccessCase, ...] = (
    AccessCase(
        'A1', _thread_listing, per_scope(lambda name: _expected_threads((*_SEED_MODEL, *_A1_ROWS), name)),
        setup=_add_rows(_A1_ROWS, pin_creation_times=True),
    ),
    AccessCase(
        'A1', _thread_pages, per_scope(_expected_pages),
        variant='page-fill', setup=_add_rows(_A1_FILL_ROWS, pin_creation_times=True),
    ),
    AccessCase('A2', _database_figures, per_scope(lambda name: _expected_database_figures(_SEED_MODEL, name))),
    AccessCase(
        'A2', _database_figures, per_scope(lambda name: _expected_database_figures((*_SEED_MODEL, *_A2_ROWS), name)),
        variant='top-n-fill', setup=_add_rows(_A2_ROWS),
    ),
    AccessCase(
        'A2', _database_size_is_deployment_wide,
        per_scope(lambda name: (len(_readable((*_SEED_MODEL, *_A2_ROWS), name)), True)),
        variant='excepted-keys', setup=_add_rows(_A2_ROWS),
    ),
    AccessCase('A3', _summary_figures, per_scope(_expected_summary_figures), setup=_add_summaries),
    AccessCase('A4', _embedding_figures, per_scope(_expected_embedding_figures), setup=_add_embedding_rows),
    AccessCase('A5', _fts_figures, per_scope(_expected_fts_figures), setup=_add_hidden_row),
    AccessCase(
        'A6', _node_count,
        per_scope(lambda name: len(READABLE[name]) + _A6_NODES * int(name in _ALICE_ONLY)), setup=_add_node_rows,
    ),
)

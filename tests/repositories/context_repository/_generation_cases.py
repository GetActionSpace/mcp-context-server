"""The context repository generation cases both backends run.

Each case takes a prepared, seeded database (:class:`~tests.repositories._access_scope_cases.ScopedDb`)
and drives the generation reads and writes of
:class:`~app.repositories.context_repository.ContextRepository` through its request
channel, reading the entry rows as the table owner through its seed channel. Every seeded
entry carries one embedding and one index-tree node and no summary.
``tests/repositories/context_repository/test_generation.py`` runs them on SQLite and
``tests/integration/postgresql/test_context_generation_mixin_postgresql.py`` on PostgreSQL.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from app import digests
from app.backends.base import MAINTENANCE
from app.ids import generate_id
from app.repositories.context_repository.records import GenerationSource
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import seed_text

_SEEDED = 'alice_private'


@dataclass(frozen=True, slots=True)
class EntryRow:
    """The columns of an entry a generation write may touch, as the table owner reads them."""

    text_content: str
    content_hash: str
    summary: str | None
    version: int
    updated_at: datetime


def _as_datetime(value: object) -> datetime:
    """Return a stored timestamp as a datetime: PostgreSQL returns one, SQLite its text form."""
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


async def read_entry(db: ScopedDb, context_id: str) -> EntryRow:
    """Read an entry's text, hash, summary, version and update time as the table owner."""
    rows = await db.fetch_all(
        'SELECT text_content, content_hash, summary, version, updated_at FROM context_entries WHERE id = ?', [context_id],
    )
    assert rows, f'entry {context_id} is missing'
    text_content, content_hash, summary, version, updated_at = rows[0]
    return EntryRow(
        text_content=str(text_content),
        content_hash=str(content_hash),
        summary=None if summary is None else str(summary),
        version=int(str(version)),
        updated_at=_as_datetime(updated_at),
    )


async def make_update_time_comparable(db: ScopedDb, context_id: str) -> None:
    """Move a SQLite entry's update time into the past, so a later write shows as an advance.

    SQLite stores whole seconds, so a write in the same second as the store would leave the
    time unchanged; PostgreSQL stores microseconds and sets the time on every update itself.
    """
    if db.seed_backend.backend_type == 'sqlite':
        await db.execute("UPDATE context_entries SET updated_at = '2020-01-01 00:00:00' WHERE id = ?", [context_id])


async def write_summary(db: ScopedDb, context_id: str, *, content_hash: str | None, summary: str | None) -> bool:
    """Write a generated summary for an entry in a transaction of its own."""
    async with db.backend.begin_transaction(access_scope=MAINTENANCE) as txn:
        return await db.repos.context.write_generation_summary(
            context_id, content_hash=content_hash, summary=summary, txn=txn,
        )


async def case_source_of_an_entry_with_outputs(db: ScopedDb) -> None:
    """The source of a seeded entry reports its text and its stored embedding and node."""
    context_id = db.ids[_SEEDED]

    source = await db.repos.context.read_generation_source(context_id, scope=MAINTENANCE)

    text = seed_text(_SEEDED)
    assert source == GenerationSource(
        text_content=text,
        source='agent',
        content_hash=digests.content_hash(text),
        has_summary=False,
        has_embeddings=True,
        node_count=1,
    )


async def case_source_of_an_entry_without_outputs(db: ScopedDb) -> None:
    """An entry stored without generated outputs reports none, and a stored summary is seen."""
    bare = await db.add_entry('bare', owner='alice', visibility='private')
    summarized = await db.add_entry('summarized', owner='bob', visibility='public', summary='Stored summary.')

    bare_source = await db.repos.context.read_generation_source(bare, scope=MAINTENANCE)
    summarized_source = await db.repos.context.read_generation_source(summarized, scope=MAINTENANCE)

    assert bare_source is not None
    assert (bare_source.has_summary, bare_source.has_embeddings, bare_source.node_count) == (False, False, 0)
    assert summarized_source is not None
    assert (summarized_source.has_summary, summarized_source.has_embeddings) == (True, False)


async def case_source_of_a_missing_entry(db: ScopedDb) -> None:
    """An id no entry carries has no source."""
    assert await db.repos.context.read_generation_source(generate_id(), scope=MAINTENANCE) is None


async def case_source_without_the_node_table(db: ScopedDb) -> None:
    """With node summaries never provisioned an entry reports no node rows."""
    await db.execute('DROP TABLE context_index_nodes')

    source = await db.repos.context.read_generation_source(db.ids[_SEEDED], scope=MAINTENANCE)

    assert source is not None
    assert (source.has_embeddings, source.node_count) == (True, 0)


async def case_summary_write_for_the_text_read(db: ScopedDb) -> None:
    """A write for the stored text sets the summary and the update time and leaves text and version alone."""
    context_id = db.ids[_SEEDED]
    await make_update_time_comparable(db, context_id)
    before = await read_entry(db, context_id)

    written = await write_summary(db, context_id, content_hash=before.content_hash, summary='Generated summary.')

    after = await read_entry(db, context_id)
    assert written is True
    assert after.summary == 'Generated summary.'
    assert (after.text_content, after.content_hash) == (before.text_content, before.content_hash)
    assert after.version == before.version
    assert after.updated_at > before.updated_at


async def case_summary_write_without_a_summary_keeps_the_stored_one(db: ScopedDb) -> None:
    """A write with no generated summary keeps the stored one and still advances the update time."""
    context_id = await db.add_entry('kept', owner='alice', visibility='private', summary='Existing summary.')
    await make_update_time_comparable(db, context_id)
    before = await read_entry(db, context_id)

    written = await write_summary(db, context_id, content_hash=before.content_hash, summary=None)

    after = await read_entry(db, context_id)
    assert written is True
    assert (after.summary, after.version) == ('Existing summary.', before.version)
    assert after.updated_at > before.updated_at


async def case_summary_write_for_changed_text_changes_nothing(db: ScopedDb) -> None:
    """A write for text other than the stored text matches no row and leaves the entry as it was."""
    context_id = db.ids[_SEEDED]
    before = await read_entry(db, context_id)

    written = await write_summary(db, context_id, content_hash=digests.content_hash('other text'), summary='Stale.')

    assert written is False
    assert await read_entry(db, context_id) == before


async def case_summary_write_for_a_missing_entry(db: ScopedDb) -> None:
    """A write for an id no entry carries matches no row."""
    written = await write_summary(db, generate_id(), content_hash=digests.content_hash('gone'), summary='Orphan.')

    assert written is False


CONTEXT_GENERATION_CASES: tuple[Callable[[ScopedDb], Awaitable[None]], ...] = (
    case_source_of_an_entry_with_outputs,
    case_source_of_an_entry_without_outputs,
    case_source_of_a_missing_entry,
    case_source_without_the_node_table,
    case_summary_write_for_the_text_read,
    case_summary_write_without_a_summary_keeps_the_stored_one,
    case_summary_write_for_changed_text_changes_nothing,
    case_summary_write_for_a_missing_entry,
)

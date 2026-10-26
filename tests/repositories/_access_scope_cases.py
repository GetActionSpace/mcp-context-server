"""Two-principal access-scope cases for the repository seams, shared by both backends.

The SQLite entry point (``tests/repositories/test_access_scope_seams.py``) and the
PostgreSQL entry point (``tests/integration/postgresql/test_access_scope_seams_postgresql.py``)
share everything here:

- The case type :class:`AccessCase`, one per seam case (unfiltered, filtered or another
  variant), each naming the database layout it runs on (``fp32`` or ``compressed``, built
  by :mod:`tests.repositories._access_scope_layouts`; on PostgreSQL every case also runs on
  the row-level security build of its layout), and the runners :func:`run_case` and
  :func:`run_case_with_scope`.
- The seed: :data:`SEED_ROWS`, eight entries owned by alice and bob covering every
  visibility and grant arrangement the access model distinguishes and every stored
  generation job state, written by :func:`seed_access_rows` through the repositories, with
  grants through the raw-SQL ``insert_grant`` helper and jobs through a raw INSERT, because
  the seed channel never passes through the seams under test.
- The two channels of a :class:`ScopedDb`. Cases call the seam under test through the
  request channel (``backend`` and ``repos``). Every row a case does not create through
  that seam (the seed, the rows of a case's ``setup`` and every child row), every raw
  statement and every expectation read goes through the seed channel (``seed_backend``
  and ``seed``), which writes and reads as the table owner, so no seed row passes through
  a request transaction that a database policy would filter. On a single-role database the
  seed channel is a second repository container over the request backend.
- The scopes the cases run as (:data:`SCOPES`) and the seed rows each scope reads, writes
  and owns (:data:`READABLE`, :data:`WRITABLE`, :data:`OWNED`, in seed order), with the
  visibility and ownership each scope reads back from them (:func:`expected_read_back`).

Adding a case: write an invoker that calls the seam as the given scope on the seeded
database and returns a comparable observable, translating entry ids to labels with
:meth:`ScopedDb.labels_of`; give the observable each scope name must produce; add the
:class:`AccessCase` to its seam group's module (``_access_scope_cases_<group>.py``), whose
cases :mod:`tests.repositories._access_scope_registry` collects into ``CASES``. Group
modules import from this module and never the other way round. Every (case, scope) pair
runs as its own test on its own freshly seeded database, under the test id
``<case id>[-<variant>][-filtered]-<scope>``, so ``pytest -k S1`` selects every variant and
scope of case S1 on both backends. A case that pins a behavior the backends differ in by
design names its backend and runs on that entry point only.

Rows beyond the seed (page-fill rows that out-rank the visible ones, hidden threads or tags
with higher counts, a second turn in a thread) come from the case's ``setup`` hook, which
runs on the freshly seeded database before the invoker and adds rows through
:meth:`ScopedDb.add_entry`, raw statements through :meth:`ScopedDb.execute` and further
child rows through the seed repositories. Every vector lies in one plane at an angle from
:data:`QUERY_VECTOR`, so the nearest-first order of any set of rows is their ascending
angle order under both the fp32 L2 distance and the compressed inner product. Seed rows sit
at :data:`SEED_ANGLE_START` plus :data:`SEED_ANGLE_STEP` per position in :data:`SEED_ROWS`,
so a row added below :data:`SEED_ANGLE_START` out-ranks every seed row. Every seed row
mentions :data:`SEED_KEYWORD`, so a row whose text repeats the keyword ranks above the seed
in full-text search.
"""

import json
import math
import sqlite3
import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import pytest
from _pytest.mark.structures import ParameterSet

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.compression.factory import get_cached_compression_provider
from app.repositories import RepositoryContainer
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from tests.helpers import insert_grant

if TYPE_CHECKING:
    import asyncpg

type Layout = Literal['fp32', 'compressed']
type BackendType = Literal['sqlite', 'postgresql']
type Visibility = Literal['private', 'public']
type Source = Literal['user', 'agent']
type JobStatus = Literal['pending', 'running', 'failed']
type Invoker = Callable[[ScopedDb, Scope], Awaitable[object]]
type Setup = Callable[[ScopedDb], Awaitable[None]]

# Case ids of the two-principal suite: context reads and probes (S), ranked search (R),
# dedup and the fresh insert (K), updates (W), deletes and owner-scoped listings (X),
# aggregates (A), the grant seams (G) and the generation job seams (J).
KNOWN_CASE_IDS = frozenset({
    'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8',
    'R1', 'R2', 'R3', 'R4',
    'K1', 'K2', 'K3', 'K4', 'K5',
    'W1', 'W2', 'W3', 'W4', 'W5',
    'X1', 'X2', 'X3',
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
    'G1', 'G2', 'G3', 'G4', 'G5',
    'J1', 'J2', 'J3', 'J4',
})

EMBEDDING_DIM = 128
SEED_THREAD = 'access-seams'
SEED_SOURCE: Source = 'agent'
SEED_KEYWORD = 'lattice'
TEAM_GROUP = 'team-x'
SEED_ANGLE_START = 0.6
SEED_ANGLE_STEP = 0.1

# A 1x1 PNG, the smallest image attachment the image repository stores.
_PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

# The scopes the cases run as. The cases set no publish role, so every writing scope may
# publish and write, exactly as the write tools resolve it; under row-level security the
# database enforces both rights, and the access predicates never read them. The anonymous
# scope is the read-only principal's: no groups, no publish right and no write right.
SCOPES: dict[str, Scope] = {
    'alice': AccessScope('alice', frozenset(), may_publish=True, may_write=True),
    'bob': AccessScope('bob', frozenset(), may_publish=True, may_write=True),
    'carol': AccessScope('carol', frozenset({TEAM_GROUP}), may_publish=True, may_write=True),
    'dave': AccessScope('dave', frozenset(), may_publish=True, may_write=True),
    'anonymous': AccessScope('anonymous', frozenset()),
    'system': SYSTEM_SCOPE,
}


@dataclass(frozen=True, slots=True)
class Grant:
    """One access grant on a seeded entry, recorded as granted by the entry's owner."""

    principal_type: Literal['user', 'group']
    principal_id: str
    permission: Literal['read', 'write']


@dataclass(frozen=True, slots=True)
class SeedRow:
    """One seeded entry: its label, owner, visibility, grants, whether it carries an image, and its generation job."""

    label: str
    owner: str
    visibility: Visibility
    grants: tuple[Grant, ...] = ()
    image: bool = False
    job: JobStatus | None = None


SEED_ROWS: tuple[SeedRow, ...] = (
    SeedRow('alice_private', 'alice', 'private', image=True, job='pending'),
    SeedRow('alice_public', 'alice', 'public', image=True, job='failed'),
    SeedRow('alice_private_bob_read', 'alice', 'private', grants=(Grant('user', 'bob', 'read'),), job='running'),
    SeedRow('alice_private_team_read', 'alice', 'private', grants=(Grant('group', TEAM_GROUP, 'read'),), job='failed'),
    SeedRow('alice_private_bob_write', 'alice', 'private', grants=(Grant('user', 'bob', 'write'),), job='pending'),
    SeedRow('alice_public_carol_write', 'alice', 'public', grants=(Grant('user', 'carol', 'write'),), job='running'),
    SeedRow('alice_public_dave_read', 'alice', 'public', grants=(Grant('user', 'dave', 'read'),)),
    SeedRow('bob_private', 'bob', 'private', image=True, job='failed'),
)
SEED_LABELS: tuple[str, ...] = tuple(row.label for row in SEED_ROWS)

# The seed rows each scope may read: its own rows, public rows and rows granted to it or to
# one of its groups; a write grant implies read. The system scope reads everything.
READABLE: dict[str, tuple[str, ...]] = {
    'alice': (
        'alice_private', 'alice_public', 'alice_private_bob_read', 'alice_private_team_read',
        'alice_private_bob_write', 'alice_public_carol_write', 'alice_public_dave_read',
    ),
    'bob': (
        'alice_public', 'alice_private_bob_read', 'alice_private_bob_write', 'alice_public_carol_write',
        'alice_public_dave_read', 'bob_private',
    ),
    'carol': ('alice_public', 'alice_private_team_read', 'alice_public_carol_write', 'alice_public_dave_read'),
    'dave': ('alice_public', 'alice_public_carol_write', 'alice_public_dave_read'),
    'anonymous': ('alice_public', 'alice_public_carol_write', 'alice_public_dave_read'),
    'system': SEED_LABELS,
}

# The seed rows each scope may modify: its own rows and rows write-granted to it or to one of
# its groups. A read grant, also on a public row, grants nothing here.
WRITABLE: dict[str, tuple[str, ...]] = {
    'alice': READABLE['alice'],
    'bob': ('alice_private_bob_write', 'bob_private'),
    'carol': ('alice_public_carol_write',),
    'dave': (),
    'anonymous': (),
    'system': SEED_LABELS,
}

# The seed rows each scope owns: the only rows it may delete or change the visibility of.
OWNED: dict[str, tuple[str, ...]] = {
    'alice': READABLE['alice'],
    'bob': ('bob_private',),
    'carol': (),
    'dave': (),
    'anonymous': (),
    'system': SEED_LABELS,
}

# The expected rows per access mode. The system scope's predicate is empty, so it admits
# every row in every mode.
VISIBILITY_TABLES: dict[AccessMode, dict[str, tuple[str, ...]]] = {
    AccessMode.READ: READABLE,
    AccessMode.WRITE: WRITABLE,
    AccessMode.OWNER: OWNED,
}


def vector_at(angle: float) -> list[float]:
    """Return the unit vector at ``angle`` radians from :data:`QUERY_VECTOR`.

    Args:
        angle: Angle from the query vector, between 0 and pi/2.

    Returns:
        An :data:`EMBEDDING_DIM`-dimensional unit vector in the plane of the first two axes.
    """
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector


QUERY_VECTOR = vector_at(0.0)


def seed_text(label: str) -> str:
    """Return the default text of an entry: a heading plus a sentence naming :data:`SEED_KEYWORD`.

    Args:
        label: The entry's label.

    Returns:
        Markdown text unique to the label.
    """
    return f'# {label}\n\nSeeded entry {label} of the access scope cases. It mentions the {SEED_KEYWORD} keyword.\n'


def per_scope[T](build: Callable[[str], T]) -> dict[str, T]:
    """Return the expectation of every scope a case runs as, the system scope included.

    Args:
        build: Builds the expected observable for one scope name.

    Returns:
        ``build(scope_name)`` keyed by every scope name in :data:`SCOPES`.
    """
    return {scope_name: build(scope_name) for scope_name in SCOPES}


def per_request_scope[T](build: Callable[[str], T]) -> dict[str, T]:
    """Return the expectation of every request scope a case runs as, leaving out the system scope.

    Args:
        build: Builds the expected observable for one scope name.

    Returns:
        ``build(scope_name)`` keyed by the name of every :class:`AccessScope` in :data:`SCOPES`.
    """
    return {scope_name: build(scope_name) for scope_name, scope in SCOPES.items() if isinstance(scope, AccessScope)}


def in_seed_order(labels: Iterable[str]) -> tuple[str, ...]:
    """Sort labels by their position in :data:`SEED_ROWS`, with labels added by cases last by name.

    Args:
        labels: Entry labels.

    Returns:
        The labels in seed order.
    """
    position = {label: index for index, label in enumerate(SEED_LABELS)}
    return tuple(sorted(labels, key=lambda label: (position.get(label, len(SEED_LABELS)), label)))


def _numbered_placeholders(sql: str) -> str:
    """Rewrite each ``?`` placeholder as the next PostgreSQL ``$n``."""
    first, *rest = sql.split('?')
    return first + ''.join(f'${index}{part}' for index, part in enumerate(rest, start=1))


def _plain(value: object) -> object:
    """Return UUID values as the 32-character hex strings the repositories use."""
    return value.hex if isinstance(value, uuid.UUID) else value


@dataclass(slots=True)
class ScopedDb:
    """A seeded database of one layout, its request and seed channels and the label of every entry.

    Attributes:
        backend: The request backend, which the seams under test run on.
        repos: The request channel: the repository container over the request backend.
        seed_backend: The backend that writes and reads as the table owner; the request
            backend itself on a single-role database.
        seed: The seed channel: the repository container over the seed backend, which
            writes every seed row and child row.
        layout: The database layout.
        ids: Entry id by label, for the seed and every row a case added.
    """

    backend: StorageBackend
    repos: RepositoryContainer
    seed_backend: StorageBackend
    seed: RepositoryContainer
    layout: Layout
    ids: dict[str, str] = field(default_factory=dict[str, str])

    def labels_of(self, context_ids: Iterable[str]) -> list[str]:
        """Translate entry ids to labels, keeping their order.

        Args:
            context_ids: Entry ids.

        Returns:
            The labels; an id no row was added under shows as ``<unlabeled <id>>``.
        """
        by_id = {context_id: label for label, context_id in self.ids.items()}
        return [by_id.get(context_id, f'<unlabeled {context_id}>') for context_id in context_ids]

    async def add_entry(
        self,
        label: str,
        *,
        owner: str,
        visibility: Visibility,
        thread_id: str = SEED_THREAD,
        source: Source = SEED_SOURCE,
        text: str | None = None,
        summary: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
        angle: float | None = None,
        image: bool = False,
        index_node: bool = False,
        grants: Sequence[Grant] = (),
    ) -> str:
        """Store one entry with its children and grants through the seed channel, and record its label.

        Args:
            label: Unique label of the entry.
            owner: Principal stamped as the owner.
            visibility: The entry's visibility.
            thread_id: Thread of the entry.
            source: Source of the entry.
            text: Entry text; defaults to :func:`seed_text` of the label.
            summary: Stored summary; none by default.
            tags: Tags; default ``seed`` and ``owner-<owner>``.
            metadata: Metadata; default ``author`` (the owner), ``label`` and ``corpus: seed``.
            angle: Store one embedding at this angle from :data:`QUERY_VECTOR`; no embedding when None.
            image: Attach one image and store the entry as multimodal.
            index_node: Store one index-tree node spanning the text.
            grants: Grants on the entry, recorded as granted by the owner.

        Returns:
            The entry id.

        Raises:
            ValueError: If the label is taken or the store merged into an existing entry.
        """
        if label in self.ids:
            raise ValueError(f'label {label!r} is already in use')
        body = text if text is not None else seed_text(label)
        entry_metadata = metadata if metadata is not None else {'author': owner, 'label': label, 'corpus': 'seed'}
        context_id, merged = await self.seed.context.store_with_deduplication(
            thread_id=thread_id,
            source=source,
            content_type='multimodal' if image else 'text',
            text_content=body,
            scope=AccessScope(owner, frozenset()),
            visibility=visibility,
            metadata=json.dumps(entry_metadata),
            summary=summary,
        )
        if merged:
            raise ValueError(f'entry {label!r} merged into the latest entry of its thread and source')
        self.ids[label] = context_id

        await self.seed.tags.store_tags(
            context_id, list(tags if tags is not None else ('seed', f'owner-{owner}')), scope=MAINTENANCE,
        )
        if image:
            await self.seed.images.store_images(
                context_id, [{'data': _PNG_BASE64, 'mime_type': 'image/png'}], scope=MAINTENANCE,
            )
        if angle is not None:
            await self._store_embedding(context_id, body, vector_at(angle))
        if index_node:
            node = IndexNodeRow(
                node_id=label, level=1, ordinal=1, title=label,
                node_summary=f'Summary of {label}.', char_start=0, char_end=len(body),
            )
            await self.seed.index_nodes.replace_nodes_for_context(context_id, [node], scope=MAINTENANCE)
        for grant in grants:
            await insert_grant(
                self.seed_backend, context_id, grant.principal_type, grant.principal_id, grant.permission, owner,
            )
        return context_id

    async def _store_embedding(self, context_id: str, text: str, vector: list[float]) -> None:
        """Store one chunk embedding through the seed channel, compressed on the compressed layout."""
        payload: bytes | None = None
        if self.layout == 'compressed':
            provider = await get_cached_compression_provider()
            payload = provider.encode_sync(np.asarray([vector], dtype=np.float32))
        chunk = ChunkEmbedding(embedding=vector, start_index=0, end_index=len(text), payload=payload)
        await self.seed.embeddings.store_chunked(context_id, [chunk], model=get_settings().embedding.model, scope=MAINTENANCE)

    async def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[tuple[object, ...]]:
        """Run one read statement through the seed channel on either backend.

        Write ``sql`` with ``?`` placeholders; on PostgreSQL each ``?`` becomes the next
        ``$n``. Text that already carries PostgreSQL placeholders, such as an access
        predicate, passes through unchanged when it holds no ``?``.

        Args:
            sql: The statement.
            params: Values to bind.

        Returns:
            The rows as tuples, with UUID values as 32-character hex strings.
        """
        if self.seed_backend.backend_type == 'sqlite':

            def _fetch_sqlite(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
                return [tuple(row) for row in conn.execute(sql, tuple(params)).fetchall()]

            return await self.seed_backend.execute_read(_fetch_sqlite, access_scope=MAINTENANCE)

        native_sql = _numbered_placeholders(sql)

        async def _fetch_postgresql(conn: 'asyncpg.Connection') -> list[tuple[object, ...]]:
            rows = await conn.fetch(native_sql, *params)
            return [tuple(_plain(value) for value in row) for row in rows]

        return await self.seed_backend.execute_read(_fetch_postgresql, access_scope=MAINTENANCE)

    async def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        """Run one write statement through the seed channel on either backend, such as backdating or re-publishing a row.

        Args:
            sql: The statement, with ``?`` placeholders as in :meth:`fetch_all`.
            params: Values to bind.
        """
        if self.seed_backend.backend_type == 'sqlite':

            def _execute_sqlite(conn: sqlite3.Connection) -> None:
                conn.execute(sql, tuple(params))

            await self.seed_backend.execute_write(_execute_sqlite, access_scope=MAINTENANCE)
            return

        native_sql = _numbered_placeholders(sql)

        async def _execute_postgresql(conn: 'asyncpg.Connection') -> None:
            await conn.execute(native_sql, *params)

        await self.seed_backend.execute_write(_execute_postgresql, access_scope=MAINTENANCE)

    async def labels_selected_by(self, scope: Scope, mode: AccessMode = AccessMode.READ) -> tuple[str, ...]:
        """Return the labels of the entries the access predicate admits for ``scope`` in ``mode``.

        Args:
            scope: The scope.
            mode: The access mode.

        Returns:
            The admitted labels in seed order.
        """
        predicate = build_access_predicate(
            scope, mode=mode, backend_type=self.seed_backend.backend_type, outer='context_entries',
        )
        rows = await self.fetch_all(f'SELECT id FROM context_entries{predicate.where_clause()}', predicate.params)
        return in_seed_order(self.labels_of(str(row[0]) for row in rows))


async def _add_job(db: ScopedDb, label: str, status: JobStatus) -> None:
    """Store a generation job of ``status`` on the entry through the seed channel; a running job holds a claim."""
    attempts, claim_token = (1, uuid.uuid4().hex) if status == 'running' else (0, None)
    await db.execute(
        'INSERT INTO generation_jobs (context_entry_id, status, attempts, claim_token) VALUES (?, ?, ?, ?)',
        (db.ids[label], status, attempts, claim_token),
    )


async def seed_access_rows(db: ScopedDb) -> None:
    """Store :data:`SEED_ROWS` in order, each with an embedding, tags, an index node, its grants and its job.

    Args:
        db: The database to seed.
    """
    for position, row in enumerate(SEED_ROWS):
        await db.add_entry(
            row.label,
            owner=row.owner,
            visibility=row.visibility,
            angle=SEED_ANGLE_START + SEED_ANGLE_STEP * position,
            image=row.image,
            index_node=True,
            grants=row.grants,
        )
        if row.job is not None:
            await _add_job(db, row.label, row.job)


@dataclass(frozen=True, slots=True)
class AccessCase:
    """One seam case of the two-principal suite.

    Attributes:
        case_id: The case id (``S1``, ``R4``, ``K1`` and so on); every test id of the case
            starts with it, so ``-k <case id>`` selects all of its variants and scopes.
        invoker: Calls the seam as the given scope on the seeded database and returns a
            comparable observable, with entry ids translated to labels.
        expected: The observable each scope must produce, keyed by scope name in
            :data:`SCOPES`; one test runs per key, and ``system`` is a key only for seams
            that accept the system scope.
        filtered: Whether the invoker passes client filters alongside the scope.
        layout: The database layout the case runs on; ``fp32`` cases need sqlite-vec on SQLite.
        variant: Tells several cases of one case id apart in their test ids, such as
            ``page-fill``, ``adversarial`` or ``owner-mode``.
        setup: Adds the rows the case needs beyond the seed, before the invoker runs.
        backend: Runs the case on this backend's entry point only, for a behavior the two
            backends differ in by design; None runs it on both.
        reports_write_access: The observable reports whether the scope may modify or owns
            each row it reads, which the seam computes with the application predicate in the
            statement's select list. The database's own answer decides only which rows the
            scope reads, so the run that leaves the database as the only filter skips it.
    """

    case_id: str
    invoker: Invoker
    expected: Mapping[str, object]
    filtered: bool = False
    layout: Layout = 'compressed'
    variant: str = ''
    setup: Setup | None = None
    backend: BackendType | None = None
    reports_write_access: bool = False

    @property
    def param_id(self) -> str:
        """The case part of the test id: the case id, the variant and ``filtered`` when set."""
        parts = (self.case_id, self.variant, 'filtered' if self.filtered else '')
        return '-'.join(part for part in parts if part)


def case_params(
    cases: Sequence[AccessCase],
    layout: Layout,
    *,
    backend: BackendType,
    marks: Sequence[pytest.MarkDecorator] = (),
) -> list[ParameterSet]:
    """Build one ``(case, scope_name)`` parameter set per case of ``layout`` and scope it names.

    Args:
        cases: The registered cases.
        layout: The layout the parametrized test seeds.
        backend: The backend of the entry point; a case restricted to the other backend is left out.
        marks: Marks applied to every parameter set, such as a skip for a missing extension.

    Returns:
        The parameter sets, with ids ``<param id>-<scope name>``.
    """
    return [
        pytest.param(case, scope_name, id=f'{case.param_id}-{scope_name}', marks=tuple(marks))
        for case in cases
        if case.layout == layout and case.backend in (None, backend)
        for scope_name in case.expected
    ]


def registry_problems(cases: Sequence[AccessCase]) -> list[str]:
    """List what makes a registry unusable: unknown case ids or scope names, duplicate test ids.

    Args:
        cases: The registered cases.

    Returns:
        One message per problem, empty for a well-formed registry.
    """
    problems: list[str] = []
    test_ids: set[str] = set()
    for case in cases:
        if case.case_id not in KNOWN_CASE_IDS:
            problems.append(f'{case.case_id}: unknown case id')
        if not case.expected:
            problems.append(f'{case.case_id}: no expected scope')
        unknown_scopes = sorted(set(case.expected) - set(SCOPES))
        if unknown_scopes:
            problems.append(f'{case.case_id}: unknown scope names {unknown_scopes}')
        for scope_name in case.expected:
            test_id = f'{case.param_id}-{scope_name}'
            if test_id in test_ids:
                problems.append(f'{case.case_id}: duplicate test id {test_id}')
            test_ids.add(test_id)
    return problems


async def run_case(db: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Run one case as one scope on a freshly seeded database and compare the observable.

    Args:
        db: The freshly seeded database of the case's layout.
        case: The case.
        scope_name: The scope to run as, a key of ``case.expected``.
    """
    await run_case_with_scope(db, case, scope_name, scope=SCOPES[scope_name])


async def run_case_with_scope(db: ScopedDb, case: AccessCase, scope_name: str, *, scope: Scope) -> None:
    """Run one case with its seams called as ``scope`` and compare the observable with that of ``scope_name``.

    :func:`run_case` passes the scope ``scope_name`` names. The row-level security run
    passes the system scope, which renders no application predicate, on a request backend
    that binds the identity of ``scope_name`` in every transaction, so the database alone
    must produce what the scope observes.

    Args:
        db: The freshly seeded database of the case's layout.
        case: The case.
        scope_name: The scope whose observable is expected, a key of ``case.expected``.
        scope: The scope the invoker calls the seams as.
    """
    if case.setup is not None:
        await case.setup(db)
    observed = await case.invoker(db, scope)
    expected = case.expected[scope_name]
    assert observed == expected, f'{case.param_id} as {scope_name}: observed {observed!r}, expected {expected!r}'


def expected_visibility() -> dict[tuple[AccessMode, str], tuple[str, ...]]:
    """Return :data:`READABLE`, :data:`WRITABLE` and :data:`OWNED` keyed by mode and scope name.

    Returns:
        The seed labels each scope is expected to read, write and own.
    """
    return {(mode, scope_name): table[scope_name] for mode, table in VISIBILITY_TABLES.items() for scope_name in SCOPES}


def expected_read_back(scope_name: str) -> dict[str, tuple[str, bool]]:
    """Return the visibility and ownership a scope reads back from each seed row it may read.

    Args:
        scope_name: The scope, a key of :data:`SCOPES`.

    Returns:
        ``(visibility, is_owner)`` keyed by the label of every row in :data:`READABLE` for
        the scope: the row's stored visibility, and whether the scope owns it.
    """
    visibility = {row.label: row.visibility for row in SEED_ROWS}
    return {label: (visibility[label], label in OWNED[scope_name]) for label in READABLE[scope_name]}


async def observe_predicate_visibility(db: ScopedDb) -> dict[tuple[AccessMode, str], tuple[str, ...]]:
    """Return the seed labels the access predicate admits for every mode and scope.

    Args:
        db: The seeded database.

    Returns:
        The admitted labels keyed by mode and scope name, in the shape of :func:`expected_visibility`.
    """
    return {
        (mode, scope_name): await db.labels_selected_by(scope, mode)
        for mode in VISIBILITY_TABLES
        for scope_name, scope in SCOPES.items()
    }

"""Two-principal cases of the update statements and the in-transaction write gate (S6, W1-W6).

Every update statement re-asserts the caller's access inside the statement itself: a
content edit requires the caller to own the row or hold a write grant on it, and a
visibility change requires the caller to own it. A row the caller may not modify behaves
exactly like an absent one: the statement reports no row, writes nothing, and a stale
compare-and-set version never surfaces as a version conflict. Each case attempts the call
on every seed row and on an absent id, then observes per row what the call reported and,
for the writes, what the rows hold afterwards through raw SQL.

- S6: ``entry_exists`` reports exactly the rows the caller may modify, outside and inside
  a transaction.
- W1: ``update_context_entry`` without a visibility change edits exactly the rows the
  caller may modify; with a stale version it raises a version conflict for those rows and
  reports the rest as absent.
- W2: ``update_context_entry`` with a visibility change, alone or together with text,
  touches exactly the rows the caller owns; a write grant is not enough. Under row-level
  security alone (:data:`RLS_UPDATE_CASES`), a caller without the publish right who sets
  its private rows public with new text is refused by the database's guard trigger, and
  those rows keep their visibility and their text.
- W3: ``patch_metadata`` patches exactly the rows the caller may modify.
- W4: ``touch_updated_at`` and ``update_content_type`` match exactly the rows the caller
  may modify, and only those rows change their content type.
- W5: ``get_content_type`` reads the content type of exactly the rows the caller may
  modify, outside and inside a transaction.
- W6: ``has_summary`` reports whether a summary is stored for exactly the rows the caller
  may modify, outside and inside a transaction.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import asyncpg

from app.access_scope import AccessScope
from app.access_scope import Scope
from app.repositories.context_repository.records import VersionConflictError
from tests.repositories._access_scope_cases import OWNED
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import WRITABLE
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import per_request_scope
from tests.repositories._access_scope_cases import per_scope
from tests.repositories._access_scope_cases_reads import ABSENT_ID
from tests.repositories._access_scope_cases_reads import ABSENT_LABEL

type Outcome = tuple[str, object]
type WriteObservable = tuple[tuple[Outcome, ...], tuple[str, ...]]

EDITED_TEXT = 'Edited by the update cases.'
STALE_VERSION = 99
# The outcome of a target whose update the database's publish gate refused.
PUBLISH_REFUSED = 'refused by the publish gate'
_SEED_VISIBILITY = {row.label: row.visibility for row in SEED_ROWS}
_SEED_CONTENT_TYPE = {row.label: 'multimodal' if row.image else 'text' for row in SEED_ROWS}
_OTHER_CONTENT_TYPE = {'text': 'multimodal', 'multimodal': 'text'}
_TARGET_LABELS = (*SEED_LABELS, ABSENT_LABEL)


def _id_of(db: ScopedDb, label: str) -> str:
    """Return the id of a seed row, or the absent id for the absent label."""
    return db.ids.get(label, ABSENT_ID)


async def _column_by_label(db: ScopedDb, column: str) -> dict[str, object]:
    """Read one column of every seed row through raw SQL, keyed by label."""
    rows = await db.fetch_all(f'SELECT id, {column} FROM context_entries')
    by_id = {str(row[0]): row[1] for row in rows}
    return {label: by_id[db.ids[label]] for label in SEED_LABELS}


def _decoded(metadata: object) -> dict[str, object]:
    """Decode a raw metadata column value, which either backend may return as JSON text."""
    decoded = json.loads(metadata) if isinstance(metadata, str) else metadata
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


async def _labels_with_text(db: ScopedDb, text: str) -> tuple[str, ...]:
    """Return the seed labels whose stored text equals ``text``, in seed order."""
    texts = await _column_by_label(db, 'text_content')
    return tuple(label for label in SEED_LABELS if texts[label] == text)


async def _exists_each(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Ask ``entry_exists`` about every seed row and the absent id as the scope."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        exists = await db.repos.context.entry_exists(_id_of(db, label), scope=scope)
        outcomes.append((label, exists))
    return tuple(outcomes)


async def _exists_each_in_transaction(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Ask ``entry_exists`` about every target on a transaction connection as the scope."""
    async with db.backend.begin_transaction(access_scope=scope) as txn:
        outcomes: list[Outcome] = []
        for label in _TARGET_LABELS:
            exists = await db.repos.context.entry_exists(_id_of(db, label), scope=scope, txn=txn)
            outcomes.append((label, exists))
        return tuple(outcomes)


def _expected_writable_flags(scope_name: str) -> tuple[Outcome, ...]:
    """True for every row the scope may modify, False for every other target."""
    return tuple((label, label in WRITABLE[scope_name]) for label in _TARGET_LABELS)


async def _edit_text(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Replace the text of every target under the captured version, then report the edited rows."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        success, fields = await db.repos.context.update_context_entry(
            _id_of(db, label), text_content=EDITED_TEXT, expected_version=0, scope=scope,
        )
        outcomes.append((label, (success, tuple(fields))))
    return tuple(outcomes), await _labels_with_text(db, EDITED_TEXT)


def _expected_text_edit(scope_name: str) -> WriteObservable:
    """Every row the scope may modify is edited; every other target reports no row."""
    outcomes = tuple(
        (label, (True, ('text_content',)) if label in WRITABLE[scope_name] else (False, ()))
        for label in _TARGET_LABELS
    )
    return outcomes, WRITABLE[scope_name]


async def _edit_text_with_stale_version(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Replace the text of every target under a stale version, then report the edited rows."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        try:
            success, _ = await db.repos.context.update_context_entry(
                _id_of(db, label), text_content=EDITED_TEXT, expected_version=STALE_VERSION, scope=scope,
            )
        except VersionConflictError:
            outcomes.append((label, 'conflict'))
        else:
            outcomes.append((label, 'updated' if success else 'not found'))
    return tuple(outcomes), await _labels_with_text(db, EDITED_TEXT)


def _expected_stale_version(scope_name: str) -> WriteObservable:
    """A row the scope may modify conflicts; every other target is not found; nothing is edited."""
    outcomes = tuple(
        (label, 'conflict' if label in WRITABLE[scope_name] else 'not found') for label in _TARGET_LABELS
    )
    return outcomes, ()


async def _publish(db: ScopedDb, scope: Scope, *, with_text: bool) -> WriteObservable:
    """Set every target public, optionally replacing its text too, then report each row's visibility."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        success, fields = await db.repos.context.update_context_entry(
            _id_of(db, label),
            text_content=EDITED_TEXT if with_text else None,
            visibility='public',
            scope=scope,
        )
        outcomes.append((label, (success, tuple(fields))))
    return tuple(outcomes), await _visibility_and_text_states(db)


async def _visibility_and_text_states(db: ScopedDb) -> tuple[str, ...]:
    """Return each seed row's visibility and whether its text is the edited text, in seed order."""
    visibilities = await _column_by_label(db, 'visibility')
    texts = await _column_by_label(db, 'text_content')
    return tuple(
        f'{visibilities[label]} {"edited" if texts[label] == EDITED_TEXT else "unchanged"}' for label in SEED_LABELS
    )


async def _publish_only(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Set every target public as the scope."""
    return await _publish(db, scope, with_text=False)


async def _publish_with_text(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Set every target public and replace its text in the same statement as the scope."""
    return await _publish(db, scope, with_text=True)


def _expected_publish(*, with_text: bool) -> Callable[[str], WriteObservable]:
    """Build the expectation: only owned rows report success and turn public, edited when text rides along."""
    fields = ('text_content', 'visibility') if with_text else ('visibility',)
    edited = 'edited' if with_text else 'unchanged'

    def build(scope_name: str) -> WriteObservable:
        owned = OWNED[scope_name]
        outcomes = tuple(
            (label, (True, fields) if label in owned else (False, ())) for label in _TARGET_LABELS
        )
        states = tuple(
            f'public {edited}' if label in owned else f'{_SEED_VISIBILITY[label]} unchanged' for label in SEED_LABELS
        )
        return outcomes, states

    return build


def _refusal(db: ScopedDb, label: str, error: asyncpg.exceptions.InsufficientPrivilegeError) -> object:
    """Name a refusal by the publish gate for the row of ``label``; report any other refusal as it is."""
    message = error.args[0] if error.args else ''
    if message == f'publishing context entry {uuid.UUID(hex=db.ids[label])} requires the publish role':
        return PUBLISH_REFUSED
    return error.sqlstate, message


async def _publish_without_the_publish_right(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Set every target public with new text as the scope without the publish right, then report each row's state.

    The repositories leave the publish right to the database: under row-level security the
    guard trigger refuses a visibility change to public for an identity bound without it.

    Returns:
        Each target's outcome, :data:`PUBLISH_REFUSED` for a refusal by the publish gate,
        and each seed row's visibility and whether it holds the edited text.
    """
    assert isinstance(scope, AccessScope)
    unpublishing = replace(scope, may_publish=False)
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        try:
            success, fields = await db.repos.context.update_context_entry(
                _id_of(db, label), text_content=EDITED_TEXT, visibility='public', scope=unpublishing,
            )
        except asyncpg.exceptions.InsufficientPrivilegeError as error:
            outcomes.append((label, _refusal(db, label, error)))
        else:
            outcomes.append((label, (success, tuple(fields))))
    return tuple(outcomes), await _visibility_and_text_states(db)


def _expected_refused_publish(scope_name: str) -> WriteObservable:
    """The publish gate refuses every private row the scope owns, which keeps its visibility and text.

    A public row the scope owns is already public, so setting it public changes no
    visibility and the gate lets its new text through. Every other target reports no row.

    Returns:
        The outcome of every target and the state of every seed row the scope must observe.
    """
    owned = OWNED[scope_name]
    owned_public = tuple(label for label in owned if _SEED_VISIBILITY[label] == 'public')

    def outcome(label: str) -> object:
        if label in owned_public:
            return True, ('text_content', 'visibility')
        if label in owned:
            return PUBLISH_REFUSED
        return False, ()

    outcomes = tuple((label, outcome(label)) for label in _TARGET_LABELS)
    states = tuple(
        'public edited' if label in owned_public else f'{_SEED_VISIBILITY[label]} unchanged' for label in SEED_LABELS
    )
    return outcomes, states


async def _patch_each(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Merge-patch the metadata of every target, then report the rows carrying the patched key."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        success, fields = await db.repos.context.patch_metadata(_id_of(db, label), {'patched': True}, scope=scope)
        outcomes.append((label, (success, tuple(fields))))
    metadata = await _column_by_label(db, 'metadata')
    patched = tuple(label for label in SEED_LABELS if _decoded(metadata[label]).get('patched') is True)
    return tuple(outcomes), patched


def _expected_patch(scope_name: str) -> WriteObservable:
    """Every row the scope may modify is patched; every other target reports no row."""
    outcomes = tuple(
        (label, (True, ('metadata',)) if label in WRITABLE[scope_name] else (False, ()))
        for label in _TARGET_LABELS
    )
    return outcomes, WRITABLE[scope_name]


async def _touch_and_retype(db: ScopedDb, scope: Scope) -> WriteObservable:
    """Stamp every target and flip its content type, then report each row's stored content type."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        target_type = _OTHER_CONTENT_TYPE[_SEED_CONTENT_TYPE.get(label, 'text')]
        touched = await db.repos.context.touch_updated_at(_id_of(db, label), scope=scope)
        retyped = await db.repos.context.update_content_type(_id_of(db, label), target_type, scope=scope)
        outcomes.append((label, (touched, retyped)))
    content_types = await _column_by_label(db, 'content_type')
    return tuple(outcomes), tuple(str(content_types[label]) for label in SEED_LABELS)


def _expected_touch_and_retype(scope_name: str) -> WriteObservable:
    """Both writes match every row the scope may modify, and only those rows flip their content type."""
    writable = WRITABLE[scope_name]
    outcomes = tuple((label, (label in writable, label in writable)) for label in _TARGET_LABELS)
    content_types = tuple(
        _OTHER_CONTENT_TYPE[_SEED_CONTENT_TYPE[label]] if label in writable else _SEED_CONTENT_TYPE[label]
        for label in SEED_LABELS
    )
    return outcomes, content_types


async def _content_type_each(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Read the content type of every target as the scope."""
    outcomes: list[Outcome] = []
    for label in _TARGET_LABELS:
        content_type = await db.repos.context.get_content_type(_id_of(db, label), scope=scope)
        outcomes.append((label, content_type))
    return tuple(outcomes)


async def _content_type_each_in_transaction(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Read the content type of every target on a transaction connection as the scope."""
    async with db.backend.begin_transaction(access_scope=scope) as txn:
        outcomes: list[Outcome] = []
        for label in _TARGET_LABELS:
            content_type = await db.repos.context.get_content_type(_id_of(db, label), scope=scope, txn=txn)
            outcomes.append((label, content_type))
        return tuple(outcomes)


def _expected_content_types(scope_name: str) -> tuple[Outcome, ...]:
    """The stored content type of every row the scope may modify, None for every other target."""
    return tuple(
        (label, _SEED_CONTENT_TYPE[label] if label in WRITABLE[scope_name] else None) for label in _TARGET_LABELS
    )


async def _summarize_every_row(db: ScopedDb) -> None:
    """Store a summary on every seed row through the seed channel."""
    await db.execute('UPDATE context_entries SET summary = ?', ['A stored summary.'])


async def _summary_presence_each(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Ask whether every target holds a summary as the scope, after every seed row received one."""
    await _summarize_every_row(db)
    outcomes: list[Outcome] = [
        (label, await db.repos.context.has_summary(_id_of(db, label), scope=scope)) for label in _TARGET_LABELS
    ]
    return tuple(outcomes)


async def _summary_presence_each_in_transaction(db: ScopedDb, scope: Scope) -> tuple[Outcome, ...]:
    """Ask whether every target holds a summary on a transaction connection as the scope."""
    await _summarize_every_row(db)
    async with db.backend.begin_transaction(access_scope=scope) as txn:
        outcomes: list[Outcome] = [
            (label, await db.repos.context.has_summary(_id_of(db, label), scope=scope, txn=txn)) for label in _TARGET_LABELS
        ]
        return tuple(outcomes)


def _expected_summary_presence(scope_name: str) -> tuple[Outcome, ...]:
    """True for every row the scope may modify, None for every other target."""
    return tuple((label, True if label in WRITABLE[scope_name] else None) for label in _TARGET_LABELS)


UPDATE_CASES: tuple[AccessCase, ...] = (
    AccessCase('S6', _exists_each, per_scope(_expected_writable_flags)),
    AccessCase('S6', _exists_each_in_transaction, per_scope(_expected_writable_flags), variant='in-transaction'),
    AccessCase('W1', _edit_text, per_scope(_expected_text_edit)),
    AccessCase('W1', _edit_text_with_stale_version, per_scope(_expected_stale_version), variant='stale-version'),
    AccessCase('W2', _publish_only, per_scope(_expected_publish(with_text=False))),
    AccessCase('W2', _publish_with_text, per_scope(_expected_publish(with_text=True)), variant='with-text'),
    AccessCase('W3', _patch_each, per_scope(_expected_patch)),
    AccessCase('W4', _touch_and_retype, per_scope(_expected_touch_and_retype)),
    AccessCase('W5', _content_type_each, per_scope(_expected_content_types)),
    AccessCase('W5', _content_type_each_in_transaction, per_scope(_expected_content_types), variant='in-transaction'),
    AccessCase('W6', _summary_presence_each, per_scope(_expected_summary_presence)),
    AccessCase(
        'W6', _summary_presence_each_in_transaction, per_scope(_expected_summary_presence), variant='in-transaction',
    ),
)

# The cases only row-level security runs: the database enforces the publish right, which the
# repositories leave to it, so no other run can observe a refusal.
RLS_UPDATE_CASES: tuple[AccessCase, ...] = (
    AccessCase(
        'W2', _publish_without_the_publish_right, per_request_scope(_expected_refused_publish),
        variant='without-publish-right',
    ),
)

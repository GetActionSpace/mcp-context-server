"""Tests for app/tools/batch/entry_validation.py.

Covers the visibility vocabulary both batch validators share with the
single-entry write tools: 'private' and 'public' pass, and any other value fails
the entry with one fixed message. Covers the per-entry deduplicate flag, read with
the same lax boolean rules as store_context's parameter, and the per-entry
defer_generation flag both validators read with the rules of the single-entry
parameters. Also covers the update validator's id resolution, which resolves a
prefix over the entries the caller may read.
"""

from typing import get_type_hints
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from app.access_scope import AccessScope
from app.auth.principal import RequestPrincipal
from app.ids import generate_id
from app.repositories.context_repository import ContextRepository
from app.tools.batch.entry_validation import validate_store_entry
from app.tools.batch.entry_validation import validate_update_entry
from app.tools.context.store import store_context
from app.tools.context.update import update_context
from tests.helpers import LOCAL_SCOPE

_VISIBILITY_ERROR = "visibility must be one of 'private', 'public'"
_PRINCIPAL = RequestPrincipal(principal_id='alice', groups=frozenset(), roles=frozenset())


def _store_entry(visibility: str) -> dict[str, object]:
    return {'thread_id': 'validation-thread', 'source': 'agent', 'text': 'entry body', 'visibility': visibility}


class TestValidateStoreEntryVisibility:
    """validate_store_entry accepts exactly 'private' and 'public'."""

    @pytest.mark.parametrize('visibility', ['private', 'public'])
    def test_private_and_public_accepted(self, visibility: str) -> None:
        """Both values pass and become the entry's effective visibility."""
        validated, error = validate_store_entry(_store_entry(visibility), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['visibility'] == visibility

    @pytest.mark.parametrize('visibility', ['shared', 'everyone'])
    def test_other_values_rejected(self, visibility: str) -> None:
        """Any other value fails the entry with the vocabulary message."""
        assert validate_store_entry(_store_entry(visibility), 0, _PRINCIPAL) == (None, _VISIBILITY_ERROR)


_DEDUPLICATE_ERROR = 'deduplicate must be a boolean'
# The single-entry boundary: FastMCP validates store_context's arguments with pydantic in lax mode.
_STORE_DEDUPLICATE_PARAMETER: TypeAdapter[bool] = TypeAdapter(
    get_type_hints(store_context, include_extras=True)['deduplicate'],
)


def _dedup_entry(value: object) -> dict[str, object]:
    return {'thread_id': 'validation-thread', 'source': 'agent', 'text': 'entry body', 'deduplicate': value}


class TestValidateStoreEntryDeduplicate:
    """validate_store_entry reads the per-entry deduplicate flag exactly as store_context's parameter does."""

    def test_absent_key_defaults_to_true(self) -> None:
        """An entry without the key deduplicates, as store_context does when the argument is omitted."""
        validated, error = validate_store_entry(_store_entry('private'), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['deduplicate'] is True

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (True, True), (False, False), (1, True), (0, False),
            ('true', True), ('false', False), ('1', True), ('0', False),
            ('yes', True), ('no', False), ('on', True), ('off', False),
            ('t', True), ('f', False), ('y', True), ('n', False), ('FALSE', False), ('Off', False),
        ],
    )
    def test_lax_boolean_values_accepted(self, value: object, expected: bool) -> None:
        """Every value the single-entry parameter coerces is accepted and coerced to the same boolean."""
        validated, error = validate_store_entry(_dedup_entry(value), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['deduplicate'] is expected
        assert _STORE_DEDUPLICATE_PARAMETER.validate_python(value) is expected

    @pytest.mark.parametrize('value', [2, -1, 'maybe', '', None, [], {}])
    def test_non_boolean_values_refused(self, value: object) -> None:
        """Every value the single-entry parameter refuses fails the entry with one fixed message, never by truthiness."""
        assert validate_store_entry(_dedup_entry(value), 4, _PRINCIPAL) == (None, _DEDUPLICATE_ERROR)
        with pytest.raises(ValidationError):
            _STORE_DEDUPLICATE_PARAMETER.validate_python(value)


_DEFER_ERROR = 'defer_generation must be a boolean'
# The single-entry boundaries the batch key mirrors, validated by FastMCP with pydantic in lax mode.
_STORE_DEFER_PARAMETER: TypeAdapter[bool] = TypeAdapter(
    get_type_hints(store_context, include_extras=True)['defer_generation'],
)
_UPDATE_DEFER_PARAMETER: TypeAdapter[bool] = TypeAdapter(
    get_type_hints(update_context, include_extras=True)['defer_generation'],
)
_LAX_BOOLEANS = [
    (True, True), (False, False), (1, True), (0, False), ('true', True), ('false', False),
    ('1', True), ('0', False), ('yes', True), ('no', False), ('on', True), ('off', False), ('FALSE', False),
]
_NON_BOOLEANS = [2, -1, 'maybe', '', None, [], {}]


def _deferred_store_entry(value: object) -> dict[str, object]:
    return {'thread_id': 'validation-thread', 'source': 'agent', 'text': 'entry body', 'defer_generation': value}


class TestValidateStoreEntryDeferGeneration:
    """validate_store_entry reads the per-entry defer_generation flag exactly as store_context's parameter does."""

    def test_absent_key_defaults_to_false(self) -> None:
        """An entry without the key generates before it is stored, as store_context does by default."""
        validated, error = validate_store_entry(_store_entry('private'), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['defer_generation'] is False

    @pytest.mark.parametrize(('value', 'expected'), _LAX_BOOLEANS)
    def test_lax_boolean_values_accepted(self, value: object, *, expected: bool) -> None:
        validated, error = validate_store_entry(_deferred_store_entry(value), 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert validated['defer_generation'] is expected
        assert _STORE_DEFER_PARAMETER.validate_python(value) is expected

    @pytest.mark.parametrize('value', _NON_BOOLEANS)
    def test_non_boolean_values_refused(self, value: object) -> None:
        assert validate_store_entry(_deferred_store_entry(value), 2, _PRINCIPAL) == (None, _DEFER_ERROR)
        with pytest.raises(ValidationError):
            _STORE_DEFER_PARAMETER.validate_python(value)


class TestValidateUpdateEntryDeferGeneration:
    """validate_update_entry reads the per-entry defer_generation flag exactly as update_context's parameter does."""

    @pytest.mark.asyncio
    async def test_absent_key_defaults_to_false(self) -> None:
        context_id = generate_id()
        validated, _, error = await validate_update_entry(
            {'context_id': context_id, 'text': 'new text'}, 0, MagicMock(spec=ContextRepository), scope=LOCAL_SCOPE,
        )
        assert error is None
        assert validated is not None
        assert validated['defer_generation'] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('value', 'expected'), _LAX_BOOLEANS)
    async def test_lax_boolean_values_accepted(self, value: object, *, expected: bool) -> None:
        context_id = generate_id()
        validated, _, error = await validate_update_entry(
            {'context_id': context_id, 'text': 'new text', 'defer_generation': value}, 0,
            MagicMock(spec=ContextRepository), scope=LOCAL_SCOPE,
        )
        assert error is None
        assert validated is not None
        assert validated['defer_generation'] is expected
        assert _UPDATE_DEFER_PARAMETER.validate_python(value) is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize('value', _NON_BOOLEANS)
    async def test_non_boolean_values_refused(self, value: object) -> None:
        context_id = generate_id()
        result = await validate_update_entry(
            {'context_id': context_id, 'text': 'new text', 'defer_generation': value}, 0,
            MagicMock(spec=ContextRepository), scope=LOCAL_SCOPE,
        )
        assert result == (None, context_id, _DEFER_ERROR)
        with pytest.raises(ValidationError):
            _UPDATE_DEFER_PARAMETER.validate_python(value)

    @pytest.mark.asyncio
    async def test_the_key_alone_is_no_update(self) -> None:
        """The flag changes how a text update generates; without a field to update the entry still fails."""
        context_id = generate_id()
        result = await validate_update_entry(
            {'context_id': context_id, 'defer_generation': True}, 0, MagicMock(spec=ContextRepository), scope=LOCAL_SCOPE,
        )
        assert result == (None, context_id, 'At least one field must be provided for update')


class TestValidateUpdateEntryVisibility:
    """validate_update_entry accepts exactly 'private' and 'public'."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('visibility', ['private', 'public'])
    async def test_private_and_public_accepted(self, visibility: str) -> None:
        """Both values pass and reach the validated update."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        validated, resolved_id, error = await validate_update_entry(
            {'context_id': context_id, 'visibility': visibility}, 0, repo, scope=LOCAL_SCOPE,
        )
        assert error is None
        assert resolved_id == context_id
        assert validated is not None
        assert validated['visibility'] == visibility

    @pytest.mark.asyncio
    @pytest.mark.parametrize('visibility', ['shared', 'everyone'])
    async def test_other_values_rejected(self, visibility: str) -> None:
        """Any other value fails the update with the vocabulary message."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        result = await validate_update_entry(
            {'context_id': context_id, 'visibility': visibility}, 0, repo, scope=LOCAL_SCOPE,
        )
        assert result == (None, context_id, _VISIBILITY_ERROR)


class TestValidateUpdateEntryIdResolution:
    """validate_update_entry resolves a prefix under the caller's scope."""

    @pytest.mark.asyncio
    async def test_prefix_resolves_under_the_given_scope(self) -> None:
        """The prefix lookup runs as the caller and its unique readable match becomes the entry id."""
        context_id = generate_id()
        bob = AccessScope('bob', frozenset())
        repo = MagicMock(spec=ContextRepository)
        repo.find_ids_by_prefix = AsyncMock(return_value=[context_id])

        validated, resolved_id, error = await validate_update_entry(
            {'context_id': context_id[:12], 'text': 'new body'}, 0, repo, scope=bob,
        )

        assert error is None
        assert resolved_id == context_id
        assert validated is not None
        repo.find_ids_by_prefix.assert_awaited_once_with(context_id[:12], limit=2, scope=bob)

    @pytest.mark.asyncio
    async def test_prefix_without_a_readable_match_fails_the_entry(self) -> None:
        """A prefix that matches no readable entry fails the entry with the resolver's message."""
        prefix = generate_id()[:12]
        repo = MagicMock(spec=ContextRepository)
        repo.find_ids_by_prefix = AsyncMock(return_value=[])

        result = await validate_update_entry({'context_id': prefix, 'text': 'new body'}, 0, repo, scope=LOCAL_SCOPE)

        assert result == (None, prefix, f"Invalid context_id: No context entry matches prefix '{prefix}'")


_INGEST_BLOCK: dict[str, object] = {
    'scope': 'kb:handbook',
    'source_id': 'guides/setup.md',
    'meta_digest': 'ab' * 32,
    'synced_at': '2026-10-18T12:00:00Z',
}
_SHAPE_ERROR = 'metadata.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at'
_PATCH_SHAPE_ERROR = (
    'metadata_patch.ingest must be null or an object with exactly the keys meta_digest, scope, source_id, synced_at'
)


class TestValidateStoreEntryIngestBlock:
    """validate_store_entry applies the ingest block validator to the entry's metadata."""

    def test_valid_block_accepted(self) -> None:
        """A complete valid block passes and reaches the serialized metadata."""
        entry = {'thread_id': 'kb:handbook', 'source': 'user', 'text': 'Setup guide', 'metadata': {'ingest': _INGEST_BLOCK}}
        validated, error = validate_store_entry(entry, 0, _PRINCIPAL)
        assert error is None
        assert validated is not None
        assert '"ingest"' in validated['metadata']

    @pytest.mark.parametrize(
        ('block', 'message'),
        [
            ({'scope': 'kb:handbook'}, _SHAPE_ERROR),
            (None, _SHAPE_ERROR),
            ({**_INGEST_BLOCK, 'scope': 5}, 'metadata.ingest.scope must be a string'),
            (
                {**_INGEST_BLOCK, 'source_id': 'x' * 257},
                'metadata.ingest.source_id is too long: 257 characters, maximum is 256',
            ),
        ],
        ids=['partial', 'null', 'non-string-scope', 'long-source-id'],
    )
    def test_malformed_block_fails_the_entry(self, block: object, message: str) -> None:
        """A malformed block fails the entry with the validator's message."""
        entry = {'thread_id': 'kb:handbook', 'source': 'user', 'text': 'Setup guide', 'metadata': {'ingest': block}}
        assert validate_store_entry(entry, 3, _PRINCIPAL) == (None, message)


class TestValidateUpdateEntryIngestBlock:
    """validate_update_entry applies the ingest block validator to metadata and to metadata_patch."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'fields',
        [
            {'metadata': {'ingest': _INGEST_BLOCK}},
            {'metadata_patch': {'ingest': _INGEST_BLOCK}},
            {'metadata_patch': {'ingest': None}},
        ],
        ids=['metadata-complete', 'patch-complete', 'patch-null'],
    )
    async def test_valid_forms_accepted(self, fields: dict[str, object]) -> None:
        """A complete block in either form, and a null block in a patch, pass."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        validated, resolved_id, error = await validate_update_entry(
            {'context_id': context_id, **fields}, 0, repo, scope=LOCAL_SCOPE,
        )
        assert error is None
        assert resolved_id == context_id
        assert validated is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('fields', 'message'),
        [
            ({'metadata': {'ingest': None}}, _SHAPE_ERROR),
            (
                {'metadata': {'ingest': {**_INGEST_BLOCK, 'synced_at': '2026-10-18T12:00:00'}}},
                'metadata.ingest.synced_at must be a UTC timestamp such as 2026-10-18T12:00:00Z',
            ),
            ({'metadata_patch': {'ingest': {'meta_digest': 'ab' * 32}}}, _PATCH_SHAPE_ERROR),
            (
                {'metadata_patch': {'ingest': {**_INGEST_BLOCK, 'meta_digest': 'AB' * 32}}},
                'metadata_patch.ingest.meta_digest must be 64 lowercase hexadecimal characters',
            ),
        ],
        ids=['metadata-null', 'metadata-no-zone', 'patch-partial', 'patch-uppercase-digest'],
    )
    async def test_malformed_block_fails_the_update(self, fields: dict[str, object], message: str) -> None:
        """A malformed block, or a partial block in a patch, fails the update with the validator's message."""
        context_id = generate_id()
        repo = MagicMock(spec=ContextRepository)
        result = await validate_update_entry({'context_id': context_id, **fields}, 0, repo, scope=LOCAL_SCOPE)
        assert result == (None, context_id, message)

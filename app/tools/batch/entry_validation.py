"""Per-entry validation shared by the batch store and update tools.

Each validator checks one batch entry and returns either the normalized entry
or a per-entry error message, so a batch can report every invalid entry at once.
"""

import json
from typing import Any
from typing import cast

from pydantic import TypeAdapter
from pydantic import ValidationError

from app.access_scope import AccessScope
from app.auth import RequestPrincipal
from app.auth import visibility_denied_reason
from app.digests import normalize_stored_text
from app.ids import resolve_or_normalize_id
from app.metadata_types import non_finite_metadata_error
from app.repositories.context_repository import ContextRepository
from app.settings import get_settings
from app.tools._validation import entry_boundary_error
from app.tools._validation import tag_limits_error
from app.tools._validation import validate_and_normalize_images

settings = get_settings()

_BOOL: TypeAdapter[bool] = TypeAdapter(bool)


def _defer_generation(entry: dict[str, Any]) -> tuple[bool, str | None]:
    """Read an entry's defer_generation flag as the single-entry write tools read their parameter.

    Args:
        entry: The raw caller-supplied entry dict.

    Returns:
        ``(flag, None)``, the flag False when the key is absent, or ``(False, error_message)``
        when the value is not one the single-entry parameter accepts.
    """
    if 'defer_generation' not in entry:
        return False, None
    try:
        return _BOOL.validate_python(entry['defer_generation']), None
    except ValidationError:
        return False, 'defer_generation must be a boolean'


def validate_store_entry(
    entry: dict[str, Any],
    idx: int,
    principal: 'RequestPrincipal',
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one store_context_batch entry against the single-entry contract.

    The single-entry store_context is Pydantic-typed at the tool boundary; the
    batch path takes untyped dicts, so this helper re-imposes the same
    rejections (types, the deduplicate and defer_generation flags, image validity,
    tag caps, metadata storability, the visibility enum, and the publish gate) so
    both paths accept and refuse identical inputs.

    Args:
        entry: The raw caller-supplied entry dict.
        idx: The entry's index in the caller's list (recorded in the result).
        principal: The batch's effective principal, for the publish gate.

    Returns:
        ``(validated_entry, None)`` on success -- the dict carries the
        normalized fields the transaction phase consumes, including the
        entry's EFFECTIVE visibility, its deduplicate flag (True when the
        key is absent) and its defer_generation flag (False when the key is
        absent) -- or ``(None, error_message)`` on the first failed check.
    """
    # Validate required fields
    if 'thread_id' not in entry or not entry.get('thread_id'):
        return None, 'Missing required field: thread_id'
    if 'source' not in entry or entry.get('source') not in ('user', 'agent'):
        return None, 'Missing or invalid source (must be "user" or "agent")'
    if 'text' not in entry or not entry.get('text'):
        return None, 'Missing required field: text'

    # The per-entry deduplicate flag follows store_context's lax boolean parameter
    # exactly (FastMCP validates it with pydantic in lax mode), never truthiness:
    # the string 'false' is truthy and must still turn deduplication off.
    deduplicate = True
    if 'deduplicate' in entry:
        try:
            deduplicate = _BOOL.validate_python(entry['deduplicate'])
        except ValidationError:
            return None, 'deduplicate must be a boolean'
    defer_generation, defer_error = _defer_generation(entry)
    if defer_error is not None:
        return None, defer_error

    # Clean input strings. Reject non-strings instead of str()-coercing:
    # coercion would silently persist the Python repr of a truthy dict/
    # list/number payload the Pydantic-typed single-entry store_context
    # rejects at the tool boundary (parity with the metadata/tags/images
    # rejections below).
    thread_id_raw = entry['thread_id']
    if not isinstance(thread_id_raw, str):
        return None, 'thread_id must be a string'
    text_raw = entry['text']
    if not isinstance(text_raw, str):
        return None, 'text must be a string'
    thread_id = thread_id_raw.strip()
    text = normalize_stored_text(text_raw)

    if not thread_id:
        return None, 'thread_id cannot be empty or whitespace'
    if not text:
        return None, 'text cannot be empty or whitespace'

    # Validate visibility and enforce the publish gate on the EFFECTIVE
    # value (a caller-omitted visibility that defaults to 'public' is
    # still a publish). Runs BEFORE image validation, matching the
    # single-entry store_context ordering so both paths surface the same
    # first error for an entry that fails both checks.
    entry_visibility = entry.get('visibility')
    if entry_visibility is not None and entry_visibility not in ('private', 'public'):
        return None, "visibility must be one of 'private', 'public'"
    effective_visibility: str = (
        entry_visibility
        if entry_visibility is not None
        else settings.access_control.default_visibility
    )
    visibility_denial = visibility_denied_reason(effective_visibility, principal)
    if visibility_denial is not None:
        return None, visibility_denial

    # Validate images if present. images is None when the caller omitted
    # the key OR passed an explicit None (both mean PRESERVE on a dedup
    # UPDATE); a provided list -- including [] -- means REPLACE.
    images = entry.get('images')
    images_provided = images is not None
    if images is not None and (
        not isinstance(images, list) or not all(isinstance(i, dict) for i in images)
    ):
        # Parity with the single-entry, Pydantic-typed store_context (which rejects a
        # non-list / non-object images). Without this, a dict-as-images or a list with a
        # non-dict element would reach validate_and_normalize_images and raise a raw
        # AttributeError/TypeError that aborts the whole non-atomic batch instead of
        # recording a per-entry error.
        return None, 'images must be a list of objects'
    if images:
        _, content_type_from_images, img_errors = validate_and_normalize_images(
            cast(list[dict[str, str]], images), error_mode='collect',
        )
        if img_errors:
            return None, img_errors[0]
        content_type = content_type_from_images
    else:
        content_type = 'text'

    # Validate metadata is a JSON object and tags is a list of strings.
    # A non-dict metadata breaks search/metadata_filters, and a bare-string
    # tags would be stored one character per tag.
    metadata = entry.get('metadata')
    if metadata is not None and not isinstance(metadata, dict):
        return None, 'metadata must be a JSON object'
    # Reject non-finite floats before generation (invalid JSON that
    # PostgreSQL rejects, so parity divergence + a wasted generation pass).
    if metadata is not None:
        metadata_error = non_finite_metadata_error(cast('object', metadata))
        if metadata_error is not None:
            return None, metadata_error
    tags = entry.get('tags')
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
    ):
        return None, 'tags must be a list of strings'

    # Per-entry tag count / per-tag length caps. The single-entry
    # store_context advertises both bounds in its wire schema; the shared
    # chokepoint enforces them here for parity (an over-long tag otherwise
    # stores on SQLite and aborts the PostgreSQL INSERT on idx_tags_tag
    # inside the transaction).
    tag_error = tag_limits_error(cast('list[str] | None', tags))
    if tag_error is not None:
        return None, tag_error

    # Reject an embedded NUL or unpaired UTF-16 surrogate in any user string
    # (thread_id, text, tags, metadata) before generation: PostgreSQL cannot
    # store it, so the entry would store on SQLite but hard-fail on PostgreSQL
    # after a wasted generation pass, charging the circuit breaker inside the
    # transaction. Mirrors the single-entry store_context boundary guard.
    # The same chokepoint also enforces the LENGTH caps on the values that land
    # in a PostgreSQL btree index (thread_id, and every INDEXED metadata field)
    # and the DECLARED TYPE of a typed indexed field: an oversized value, or one
    # an integer or boolean index cannot cast, aborts the PostgreSQL INSERT where
    # SQLite stores it. It also refuses a malformed reserved ingest metadata block,
    # as store_context does.
    entry_error = entry_boundary_error(
        thread_id=thread_id,
        text=text,
        tags=cast('object', tags),
        metadata=cast('object', metadata),
    )
    if entry_error is not None:
        return None, entry_error

    # Prepare validated entry. tags/images keep their None-ness so the
    # store can distinguish PRESERVE (None) from REPLACE-with-empty ([])
    # on a dedup UPDATE, matching the documented replacement contract.
    return {
        'index': idx,
        'thread_id': thread_id,
        'source': entry['source'],
        'text_content': text,
        'metadata': json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
        'content_type': content_type,
        'tags': tags,
        'images': images if images is not None else [],
        'images_provided': images_provided,
        'visibility': effective_visibility,
        'deduplicate': deduplicate,
        'defer_generation': defer_generation,
    }, None


async def validate_update_entry(
    update: dict[str, Any],
    idx: int,
    context_repo: ContextRepository,
    *,
    scope: AccessScope,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Validate one update_context_batch entry against the single-entry contract.

    The single-entry update_context is Pydantic-typed at the tool boundary; the
    batch path takes untyped dicts, so this helper re-imposes the same
    rejections (id resolution, mutual exclusivity, types, the defer_generation
    flag, image validity, tag caps, metadata storability, and the visibility
    enum) so both paths accept and refuse identical inputs. The owner-only visibility authorization runs
    later, in the existence phase, once the entry's stamped owner is known.

    Args:
        update: The raw caller-supplied update dict.
        idx: The entry's index in the caller's list (recorded in the result).
        context_repo: Context repository used to resolve id prefixes.
        scope: The caller's scope; an id prefix resolves over the entries it may read.

    Returns:
        ``(validated_update, context_id, None)`` on success, or
        ``(None, context_id, error_message)`` on the first failed check
        (``context_id`` is ``''`` when the id itself could not be resolved,
        matching the single-inline-loop error rows this replaced).
    """
    # Validate required context_id
    if 'context_id' not in update:
        return None, '', 'Missing required field: context_id'

    context_id_raw = update['context_id']
    if not isinstance(context_id_raw, str) or not context_id_raw.strip():
        return None, '', 'context_id must be a non-empty string'

    # Resolve to canonical 32-char hex (accept full or prefix)
    try:
        context_id = await resolve_or_normalize_id(context_id_raw, context_repo, scope=scope)
    except ValueError as e:
        return None, context_id_raw, f'Invalid context_id: {e}'

    defer_generation, defer_error = _defer_generation(update)
    if defer_error is not None:
        return None, context_id, defer_error

    # Validate mutual exclusivity of metadata and metadata_patch
    if update.get('metadata') is not None and update.get('metadata_patch') is not None:
        return None, context_id, 'Cannot use both metadata and metadata_patch. Use one or the other.'

    # Validate text if provided. Reject non-strings instead of
    # str()-coercing: coercion would silently persist the Python repr
    # of a dict/list/number payload the Pydantic-typed single-entry
    # update_context rejects at the tool boundary (parity with the
    # metadata/tags rejections below).
    text = update.get('text')
    if text is not None:
        if not isinstance(text, str):
            return None, context_id, 'text must be a string'
        text = normalize_stored_text(text)
        if not text:
            return None, context_id, 'text cannot be empty or whitespace'

    # Check that at least one field is provided for update
    has_update = any(
        update.get(field) is not None
        for field in ['text', 'metadata', 'metadata_patch', 'tags', 'images', 'visibility']
    )
    if not has_update:
        return None, context_id, 'At least one field must be provided for update'

    # Validate visibility (parity with the single-entry Literal-typed
    # update_context).
    visibility_field = update.get('visibility')
    if visibility_field is not None and visibility_field not in ('private', 'public'):
        return None, context_id, "visibility must be one of 'private', 'public'"

    # Validate metadata / metadata_patch are JSON objects and tags is a
    # list of strings (parity with the single-entry, Pydantic-typed
    # update_context, which rejects these).
    metadata_field = update.get('metadata')
    if metadata_field is not None and not isinstance(metadata_field, dict):
        return None, context_id, 'metadata must be a JSON object'
    metadata_patch_field = update.get('metadata_patch')
    if metadata_patch_field is not None and not isinstance(metadata_patch_field, dict):
        return None, context_id, 'metadata_patch must be a JSON object'
    tags_field = update.get('tags')
    if tags_field is not None and (
        not isinstance(tags_field, list) or not all(isinstance(t, str) for t in tags_field)
    ):
        return None, context_id, 'tags must be a list of strings'
    # Per-entry tag count / per-tag length caps (see the store batch for
    # the full rationale): the untyped batch path must reject exactly what
    # the wire-schema-bounded single-entry update_context rejects.
    tag_error = tag_limits_error(cast('list[str] | None', tags_field))
    if tag_error is not None:
        return None, context_id, tag_error

    # Validate images if provided
    images = update.get('images')
    if images is not None and (
        not isinstance(images, list) or not all(isinstance(i, dict) for i in images)
    ):
        # Parity with the single-entry, Pydantic-typed update_context (which rejects a
        # non-list / non-object images). Without this, a dict-as-images or a list with a
        # non-dict element would reach validate_and_normalize_images and raise a raw
        # AttributeError/TypeError that aborts the whole non-atomic batch instead of
        # recording a per-entry error.
        return None, context_id, 'images must be a list of objects'
    if images:
        _, _, img_errors = validate_and_normalize_images(
            cast(list[dict[str, str]], images), error_mode='collect',
        )
        if img_errors:
            return None, context_id, img_errors[0]

    # Reject non-finite floats in metadata (full or patch) before
    # generation: invalid JSON that PostgreSQL rejects, so parity
    # divergence + a wasted generation pass.
    for meta_value in (update.get('metadata'), update.get('metadata_patch')):
        if meta_value is not None:
            non_finite_error = non_finite_metadata_error(cast('object', meta_value))
            if non_finite_error is not None:
                return None, context_id, non_finite_error

    # Reject an embedded NUL or unpaired UTF-16 surrogate in any user string
    # (text, tags, metadata, metadata_patch) before generation, mirroring the
    # single-entry update_context boundary guard: PostgreSQL cannot store it,
    # so the update would succeed on SQLite but hard-fail on PostgreSQL after a
    # wasted generation pass, charging the circuit breaker inside the transaction.
    # The same chokepoint also enforces the length cap and, for a typed field,
    # the declared type of every INDEXED metadata field, in both the replacement
    # and the merge-patch form: an oversized value, or one an integer or boolean
    # index cannot cast, aborts the PostgreSQL UPDATE on idx_metadata_<field>
    # where SQLite stores it. It also refuses a malformed reserved ingest metadata
    # block in either form, and a partial one in the merge patch, as update_context does.
    entry_error = entry_boundary_error(
        text=text,
        tags=cast('object', tags_field),
        metadata=cast('object', metadata_field),
        metadata_patch=cast('object', metadata_patch_field),
    )
    if entry_error is not None:
        return None, context_id, entry_error

    return {
        'index': idx,
        'context_id': context_id,
        'text': text,
        'metadata': update.get('metadata'),
        'metadata_patch': update.get('metadata_patch'),
        'tags': update.get('tags'),
        'images': images,
        'visibility': visibility_field,
        'defer_generation': defer_generation,
    }, context_id, None

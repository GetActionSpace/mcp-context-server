"""Tests for the order of the database preparation phase in app.startup.database_setup.

Starting with ``ENABLE_POSTGRESQL_RLS`` off over the row-level security objects is
refused right after the compression guard, before any provisioning migration runs. The
ingest index runs right after the access-control migration, because its SQLite key names
the access columns that migration adds on an in-place upgrade, and the grant owner
migration follows it, because its index and foreign key name ``owner_id``, which the
access-control migration adds the same way. The request role's grants and then the
row-level security migration run last, once every table and column this boot provisions
exists, so the grants cover every managed table and the read view lists every column.
"""

import inspect
from types import SimpleNamespace
from typing import cast

import pytest

import app.startup.database_setup as database_setup
from app.backends import StorageBackend
from app.settings import AppSettings
from app.startup.database_setup import prepare_database

_ASYNC_STEPS = (
    'init_database',
    'handle_metadata_indexes',
    'guard_compression_disable_over_populated',
    'guard_rls_disable_over_enabled',
    'apply_semantic_search_migration',
    'apply_jsonb_merge_patch_migration',
    'apply_function_search_path_migration',
    'apply_fts_migration',
    'apply_chunking_migration',
    'apply_summary_migration',
    'apply_content_hash_migration',
    'apply_version_migration',
    'apply_access_control_migration',
    'apply_ingest_index_migration',
    'apply_grant_owner_migration',
    'apply_tag_uniqueness_migration',
    'apply_compression_migration',
    'apply_index_tree_migration',
    'validate_compression_provenance',
    '_announce_compression',
    'apply_request_role_grants',
    'apply_rls',
)


def test_the_request_role_is_a_required_keyword() -> None:
    parameter = inspect.signature(prepare_database).parameters['request_role']

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


async def _recorded_steps(
    monkeypatch: pytest.MonkeyPatch, backend: StorageBackend, settings: AppSettings,
) -> list[tuple[str, tuple[object, ...]]]:
    """Run ``prepare_database`` with every step replaced by a recorder, and return the calls in order."""
    calls: list[tuple[str, tuple[object, ...]]] = []

    def _recorder(name: str) -> object:
        async def _step(*args: object, **kwargs: object) -> None:
            calls.append((name, (*args, *kwargs.values())))

        return _step

    for name in _ASYNC_STEPS:
        monkeypatch.setattr(database_setup, name, _recorder(name))

    def _validate_pool_acquire_timeout() -> None:
        calls.append(('validate_pool_acquire_timeout', ()))

    monkeypatch.setattr(database_setup, 'validate_pool_acquire_timeout', _validate_pool_acquire_timeout)

    await prepare_database(backend, settings, request_role='mcp_app')
    return calls


@pytest.mark.asyncio
async def test_the_rls_guard_runs_early_and_the_grants_and_the_rls_migration_run_last(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cast(StorageBackend, SimpleNamespace(backend_type='postgresql'))
    settings = cast(AppSettings, SimpleNamespace())

    calls = await _recorded_steps(monkeypatch, backend, settings)

    names = [name for name, _ in calls]
    assert names.index('guard_rls_disable_over_enabled') == names.index('guard_compression_disable_over_populated') + 1
    assert names.index('guard_rls_disable_over_enabled') < names.index('apply_semantic_search_migration')
    assert names[-3:] == ['_announce_compression', 'apply_request_role_grants', 'apply_rls']
    arguments = dict(calls)
    assert arguments['guard_rls_disable_over_enabled'] == (backend, settings)
    assert arguments['apply_request_role_grants'] == (backend, settings, 'mcp_app')
    assert arguments['apply_rls'] == (backend, settings)


@pytest.mark.asyncio
async def test_the_ingest_index_is_created_right_after_the_access_control_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = cast(StorageBackend, SimpleNamespace(backend_type='sqlite'))

    calls = await _recorded_steps(monkeypatch, backend, cast(AppSettings, SimpleNamespace()))

    names = [name for name, _ in calls]
    assert names.index('apply_ingest_index_migration') == names.index('apply_access_control_migration') + 1
    assert dict(calls)['apply_ingest_index_migration'] == (backend,)


@pytest.mark.asyncio
async def test_the_grant_owner_migration_follows_the_access_columns_and_precedes_the_grants_and_rls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = cast(StorageBackend, SimpleNamespace(backend_type='postgresql'))

    calls = await _recorded_steps(monkeypatch, backend, cast(AppSettings, SimpleNamespace()))

    names = [name for name, _ in calls]
    assert names.index('apply_grant_owner_migration') == names.index('apply_ingest_index_migration') + 1
    assert names.index('apply_grant_owner_migration') < names.index('apply_tag_uniqueness_migration')
    assert names.index('apply_grant_owner_migration') < names.index('apply_request_role_grants')
    assert dict(calls)['apply_grant_owner_migration'] == (backend,)

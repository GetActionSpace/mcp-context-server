"""Caller scoping of ``get_statistics``: scoped figures and the deployment-level keys.

Every figure derived from stored entries is computed over the entries the caller may read;
the database size, the embedding storage size, the connection metrics and the configuration
blocks describe the deployment and are the same for every caller.
"""

import base64
from pathlib import Path
from typing import Any
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.startup
import app.tools
from app.access_scope import AccessScope
from tests.helpers import as_principal
from tests.tools._generation_jobs import seed_generation_jobs

# The tool functions are plain coroutines that lifespan() registers with FastMCP at startup; tests call them directly.
store_context = app.tools.store_context
get_statistics = app.tools.get_statistics

# Blocks that describe the deployment rather than the stored entries.
_CONFIGURATION_BLOCKS = ('chunking', 'reranking', 'compression')
_CONFIGURATION_KEYS = {
    'semantic_search': ('enabled', 'available', 'backend', 'model', 'dimensions'),
    'fts': ('enabled', 'available', 'language', 'backend', 'engine'),
    'summary': ('enabled', 'available', 'provider', 'model', 'min_content_length'),
    'index_tree': ('enabled',),
}


async def _store_as(
    principal_id: str,
    *,
    thread_id: str,
    text: str,
    visibility: Literal['private', 'public'],
    tags: list[str],
    with_image: bool = False,
) -> None:
    """Store one entry as the principal."""
    images = [{'data': base64.b64encode(b'img').decode('utf-8')}] if with_image else None
    with as_principal(principal_id):
        await store_context(
            thread_id=thread_id, source='agent', text=text, visibility=visibility, tags=tags, images=images,
        )


async def _statistics_as(principal_id: str) -> dict[str, Any]:
    """Return ``get_statistics`` as the principal."""
    with as_principal(principal_id):
        return cast(dict[str, Any], await get_statistics())


async def _store_alice_entries() -> None:
    """Store a private multimodal alice entry and a public alice entry in separate threads."""
    await _store_as(
        'alice', thread_id='alice-private-thread', text='alice private statistics entry', visibility='private',
        tags=['secret'], with_image=True,
    )
    await _store_as(
        'alice', thread_id='alice-public-thread', text='alice public statistics entry', visibility='public',
        tags=['open'],
    )


@pytest.mark.usefixtures('initialized_server')
class TestStatisticsScoping:
    """get_statistics computes every entry-derived figure over the caller's readable entries."""

    @pytest.mark.asyncio
    async def test_scoped_figures_follow_the_caller(self) -> None:
        """Bob counts only alice's public entry; alice counts both of hers."""
        await _store_alice_entries()

        bob = await _statistics_as('bob')
        alice = await _statistics_as('alice')

        assert {key: bob[key] for key in ('total_entries', 'total_threads', 'total_images', 'unique_tags')} == {
            'total_entries': 1, 'total_threads': 1, 'total_images': 0, 'unique_tags': 1,
        }
        assert bob['by_content_type'] == {'text': 1}
        assert bob['most_active_threads'] == [{'thread_id': 'alice-public-thread', 'count': 1}]
        assert bob['top_tags'] == [{'tag': 'open', 'count': 1}]
        assert {key: alice[key] for key in ('total_entries', 'total_threads', 'total_images', 'unique_tags')} == {
            'total_entries': 2, 'total_threads': 2, 'total_images': 1, 'unique_tags': 2,
        }
        assert alice['by_content_type'] == {'text': 1, 'multimodal': 1}

    @pytest.mark.asyncio
    async def test_generation_counts_follow_the_caller(self) -> None:
        """Each caller counts the pending and failed generation of the entries it may read; a running job is pending."""
        stored: dict[str, str] = {}
        entries: tuple[tuple[str, Literal['private', 'public']], ...] = (
            ('private', 'private'), ('public-running', 'public'), ('public-failed', 'public'),
        )
        for label, visibility in entries:
            with as_principal('alice'):
                result = await store_context(
                    thread_id='generation-statistics', source='agent', text=f'alice {label} entry', visibility=visibility,
                )
            stored[label] = result['context_id']
        repos = await app.startup.ensure_repositories()
        await seed_generation_jobs(
            repos.context.backend,
            {stored['private']: 'pending', stored['public-running']: 'running', stored['public-failed']: 'failed'},
        )

        bob = await _statistics_as('bob')
        alice = await _statistics_as('alice')

        assert bob['generation'] == {'pending': 1, 'failed': 1}
        assert alice['generation'] == {'pending': 2, 'failed': 1}

    @pytest.mark.asyncio
    async def test_deployment_level_keys_are_the_same_for_every_caller(self, temp_db_path: Path) -> None:
        """The database size, embedding storage size and configuration blocks do not depend on the caller."""
        await _store_alice_entries()

        with patch('app.tools.discovery.DB_PATH', temp_db_path):
            alice = await _statistics_as('alice')
            bob = await _statistics_as('bob')

        assert alice['total_entries'] != bob['total_entries']
        assert 'database_size_mb' in bob
        assert bob['database_size_mb'] == alice['database_size_mb']
        for key in ('embeddings_size_mb', 'embeddings_size_estimated'):
            assert bob.get(key) == alice.get(key)
        assert set(bob['connection_metrics']) == set(alice['connection_metrics'])
        for block in _CONFIGURATION_BLOCKS:
            assert bob[block] == alice[block]
        for block, keys in _CONFIGURATION_KEYS.items():
            assert {key: bob[block].get(key) for key in keys} == {key: alice[block].get(key) for key in keys}

    @pytest.mark.asyncio
    async def test_scope_reaches_every_scoped_repository_call(self) -> None:
        """Each repository call behind a scoped figure receives the caller's scope."""
        repos = await app.startup.ensure_repositories()
        settings = MagicMock()
        settings.semantic_search.enabled = True
        settings.fts.enabled = True
        settings.summary.generation_enabled = True
        settings.index_tree.node_summaries_enabled = True
        settings.embedding.generation_enabled = False
        settings.compression.enabled = False

        with (
            as_principal('bob', groups=['team-x']),
            patch('app.tools.discovery.settings', settings),
            patch('app.tools.discovery.get_embedding_provider', return_value=MagicMock()),
            patch('app.startup.get_summary_provider', return_value=MagicMock()),
            patch.object(repos.statistics, 'get_database_statistics', AsyncMock(return_value={})) as database_spy,
            patch.object(
                repos.embeddings, 'get_statistics',
                AsyncMock(return_value={
                    'backend': 'sqlite', 'total_embeddings': 0, 'total_chunks': 0,
                    'average_chunks_per_entry': 0.0, 'coverage_percentage': 0.0,
                }),
            ) as embeddings_spy,
            patch.object(repos.fts, 'is_available', AsyncMock(return_value=True)),
            patch.object(
                repos.fts, 'get_statistics',
                AsyncMock(return_value={
                    'backend': 'sqlite', 'engine': 'fts5', 'indexed_entries': 0, 'coverage_percentage': 0.0,
                }),
            ) as fts_spy,
            patch.object(
                repos.statistics, 'get_summary_statistics',
                AsyncMock(return_value={'summary_count': 0, 'coverage_percentage': 0.0}),
            ) as summary_spy,
            patch.object(repos.index_nodes, 'count_all_nodes', AsyncMock(return_value=0)) as nodes_spy,
        ):
            await get_statistics()

        expected = AccessScope('bob', frozenset({'team-x'}))
        for spy in (database_spy, embeddings_spy, fts_spy, summary_spy, nodes_spy):
            assert spy.await_args is not None
            assert spy.await_args.kwargs['scope'] == expected

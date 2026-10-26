"""Tool-level tests for navigate_context (SQLite backend): the outline tree with its summary root and
stored per-node summaries.
"""

import sqlite3
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.migrations.index_tree import apply_index_tree_migration
from app.repositories.index_node_repository import IndexNodeRow
from app.repositories.index_node_repository import StoredNodeSummaries
from app.startup import ensure_repositories
from app.tools.sharing import unshare_context
from tests.helpers import as_principal
from tests.helpers import insert_grant
from tests.tools._generation_jobs import seed_generation_jobs
from tests.tools.navigation._navigation import navigate_as_dict
from tests.tools.navigation._navigation import store_entry


class TestNavigateContext:
    """navigate_context builds the on-demand Markdown outline."""

    @pytest.mark.asyncio
    async def test_outline_tree_and_summary_root(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, '# Intro\nhello\n## Details\nmore\n')
        result = await navigate_as_dict(context_id=cid)
        assert result['context_id'] == cid
        assert result['node_count'] == 2
        root = result['root']
        assert root['node_id'] == 'root'
        # No stored summary on this short entry -> root summary mirrors None.
        assert root['summary'] is None
        intro = root['children'][0]
        assert intro['title'] == 'Intro'
        assert intro['children'][0]['title'] == 'Details'

    @pytest.mark.asyncio
    async def test_headingless_entry_yields_childless_root(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'plain text with no headings at all')
        result = await navigate_as_dict(context_id=cid)
        assert result['node_count'] == 0
        assert result['root']['children'] == []

    @pytest.mark.asyncio
    async def test_missing_entry_raises(self, nav_backend: StorageBackend) -> None:
        assert nav_backend is not None
        with pytest.raises(ToolError):
            await navigate_as_dict(context_id='0' * 32)


class TestNavigateContextScoping:
    """navigate_context outlines only entries the caller may read."""

    @pytest.mark.asyncio
    async def test_unreadable_entry_fails_like_a_missing_one(self, nav_backend: StorageBackend) -> None:
        """Another principal's private entry yields the same not-found error as an absent id."""
        hidden_id = await store_entry(nav_backend, '# Alice\nprivate\n', owner='alice')
        absent_id = '0' * 32

        with pytest.raises(ToolError) as hidden:
            await navigate_as_dict(context_id=hidden_id)
        with pytest.raises(ToolError) as absent:
            await navigate_as_dict(context_id=absent_id)

        assert str(hidden.value) == f'Context entry not found: {hidden_id}'
        assert str(absent.value) == f'Context entry not found: {absent_id}'

    @pytest.mark.asyncio
    async def test_no_node_read_for_an_unreadable_entry(self, nav_backend: StorageBackend) -> None:
        """With node summaries requested, an unreadable entry never reaches the node-summary read."""
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        hidden_id = await store_entry(nav_backend, '# Alice\nprivate\n', owner='alice')

        with (
            patch.object(repos.index_nodes, 'get_nodes_for_context', AsyncMock()) as nodes_spy,
            pytest.raises(ToolError, match='Context entry not found'),
        ):
            await navigate_as_dict(context_id=hidden_id, include_node_summaries=True)

        nodes_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_outlines_their_private_entry(self, nav_backend: StorageBackend) -> None:
        """The owner of a private entry gets its outline."""
        cid = await store_entry(nav_backend, '# Alice\nprivate\n', owner='alice')

        with as_principal('alice'):
            result = await navigate_as_dict(context_id=cid)

        assert result['root']['children'][0]['title'] == 'Alice'


async def _publish(backend: StorageBackend, context_id: str) -> None:
    """Make one stored entry public."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE context_entries SET visibility = 'public' WHERE id = ?", (context_id,))

    await backend.execute_write(_write, access_scope=MAINTENANCE)


class TestNavigateContextReadBack:
    """The response reports the entry's visibility, ownership and generation state at its top level only."""

    @pytest.mark.asyncio
    async def test_top_level_carries_generation_state(self, nav_backend: StorageBackend) -> None:
        """The response reports the entry's generation state at the top level, never on an outline node."""
        ready = await store_entry(nav_backend, '# Ready\nbody')
        pending = await store_entry(nav_backend, '# Pending\nbody', offset_seconds=1)
        failed = await store_entry(nav_backend, '# Failed\nbody', offset_seconds=2)
        await seed_generation_jobs(nav_backend, {pending: 'running', failed: 'failed'})

        results = {context_id: await navigate_as_dict(context_id=context_id) for context_id in (ready, pending, failed)}

        assert {context_id: result['generation_status'] for context_id, result in results.items()} == {
            ready: 'ready', pending: 'pending', failed: 'failed',
        }
        for result in results.values():
            assert 'generation_status' not in result['root']
            assert all('generation_status' not in child for child in result['root']['children'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize('include_node_summaries', [False, True])
    async def test_top_level_carries_visibility_and_ownership(
        self, nav_backend: StorageBackend, include_node_summaries: bool,
    ) -> None:
        """Alice owns her public entry, the default principal reads it as not owned; no node carries the fields."""
        await apply_index_tree_migration(nav_backend, force=True)
        cid = await store_entry(nav_backend, '# Alice\npublic\n## Part\nmore\n', owner='alice')
        await _publish(nav_backend, cid)

        with as_principal('alice'):
            owned = await navigate_as_dict(context_id=cid, include_node_summaries=include_node_summaries)
        read = await navigate_as_dict(context_id=cid, include_node_summaries=include_node_summaries)

        assert (owned['visibility'], owned['is_owner']) == ('public', True)
        assert (read['visibility'], read['is_owner']) == ('public', False)
        for result in (owned, read):
            assert type(result['is_owner']) is bool
            assert 'owner_id' not in result
            nodes = [result['root']]
            while nodes:
                node = nodes.pop()
                assert {'visibility', 'is_owner', 'owner_id'}.isdisjoint(node)
                nodes.extend(node['children'])


class TestNavigateNodeSummaries:
    """include_node_summaries surfaces stored per-node summaries onto descendants."""

    @pytest.mark.asyncio
    async def test_stored_node_summaries_surface(self, nav_backend: StorageBackend) -> None:
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await store_entry(nav_backend, '# Intro\nhello\n## Details\nmore body here\n')

        baseline = await navigate_as_dict(context_id=cid)
        details = baseline['root']['children'][0]['children'][0]
        assert details['node_id'] == 'intro/details'

        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='intro/details', level=2, ordinal=1, title='Details',
                node_summary='Covers the details.',
                char_start=details['char_start'], char_end=details['char_end'],
            ),
        ], scope=MAINTENANCE)

        enriched = await navigate_as_dict(context_id=cid, include_node_summaries=True)
        assert enriched['root']['children'][0]['children'][0]['summary'] == 'Covers the details.'

    @pytest.mark.asyncio
    async def test_pre_cap_stored_node_ids_reattach_by_span(self, nav_backend: StorageBackend) -> None:
        """Rows keyed by an older slug algorithm's node_id re-attach by span.

        The slug segment cap shortens the computed node_id for long headings,
        so rows stored under the pre-cap slug rules carry the uncapped id for
        the SAME section. The document is unchanged, so the stored (char_start,
        char_end) still matches the recomputed section exactly and the
        summary must surface instead of silently detaching until the next
        text-change update regenerates the rows.
        """
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        long_title = ('alpha beta gamma delta ' * 8).strip()  # slugifies past the cap
        cid = await store_entry(nav_backend, f'# {long_title}\nsection body here\n')

        baseline = await navigate_as_dict(context_id=cid)
        section = baseline['root']['children'][0]
        # The uncapped slug an older parse produced for the identical title.
        pre_cap_id = ('alpha-beta-gamma-delta-' * 8).rstrip('-')
        assert section['node_id'] != pre_cap_id  # the ids genuinely diverged

        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id=pre_cap_id, level=1, ordinal=1, title=long_title,
                node_summary='Section abstract from before the cap.',
                char_start=section['char_start'], char_end=section['char_end'],
            ),
        ], scope=MAINTENANCE)

        enriched = await navigate_as_dict(context_id=cid, include_node_summaries=True)
        node = enriched['root']['children'][0]
        assert node['summary'] == 'Section abstract from before the cap.'
        # The synthetic root keeps mirroring the entry summary (None here) even
        # though this document-spanning section shares the root's char span.
        assert enriched['root']['summary'] is None

    @pytest.mark.asyncio
    async def test_colliding_pre_cap_id_does_not_misattach_sibling_summary(
        self, nav_backend: StorageBackend,
    ) -> None:
        """A stale node_id hit with a mismatched span must not steal a sibling's summary.

        Node ids are algorithm-versioned: under the pre-cap slug rules a SHORT
        sibling's id can equal the capped prefix the current parse assigns
        to a LONGER sibling that appears first (the short one then carries an
        ordinal suffix). An unvalidated by_node_id hit would attach the short
        section's stored summary to the long section and leave the long
        section's own row unused. The reader must trust an id hit only when
        the stored row's span matches the computed node's span, falling
        through to the span index otherwise, so BOTH sections keep their own
        summaries.
        """
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        long_title = ('alpha beta gamma delta ' * 8).strip()
        # Slugifies to EXACTLY the 64-code-point capped prefix of long_title's slug.
        short_title = 'alpha beta gamma delta alpha beta gamma delta alpha beta gamma d'
        cid = await store_entry(
            nav_backend,
            f'# {long_title}\nlong body\n# {short_title}\nshort body\n',
        )

        baseline = await navigate_as_dict(context_id=cid)
        long_node, short_node = baseline['root']['children']
        capped = 'alpha-beta-gamma-delta-alpha-beta-gamma-delta-alpha-beta-gamma-d'
        assert long_node['node_id'] == capped  # the long-first sibling owns the capped base id
        assert short_node['node_id'] != capped  # the short sibling got an ordinal suffix

        pre_cap_long_id = ('alpha-beta-gamma-delta-' * 8).rstrip('-')
        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id=pre_cap_long_id, level=1, ordinal=1, title=long_title,
                node_summary='Long section abstract.',
                char_start=long_node['char_start'], char_end=long_node['char_end'],
            ),
            IndexNodeRow(
                node_id=capped, level=1, ordinal=2, title=short_title,
                node_summary='Short section abstract.',
                char_start=short_node['char_start'], char_end=short_node['char_end'],
            ),
        ], scope=MAINTENANCE)

        enriched = await navigate_as_dict(context_id=cid, include_node_summaries=True)
        enriched_long, enriched_short = enriched['root']['children']
        assert enriched_long['summary'] == 'Long section abstract.'
        assert enriched_short['summary'] == 'Short section abstract.'

    @pytest.mark.asyncio
    async def test_same_second_concurrent_update_retakes_snapshot(self, nav_backend: StorageBackend) -> None:
        """A text-change committing between the two reads forces a snapshot retake.

        SQLite CURRENT_TIMESTAMP has second granularity, so an updated_at
        probe accepts the torn old-text/new-summaries pairing whenever the
        concurrent update lands in the same wall-clock second as the prior
        write -- which a sub-second commit always does. The version token
        moves on every node-replacing write, so the probe must detect the
        update and the result must pair the new text's outline with its own
        node summaries.
        """
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await store_entry(nav_backend, '# Old\nold body\n')
        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='old', level=1, ordinal=1, title='Old',
                node_summary='Old section abstract.', char_start=0, char_end=15,
            ),
        ], scope=MAINTENANCE)

        real_get_nodes = repos.index_nodes.get_nodes_for_context
        update_committed = False

        async def racing_get_nodes(context_id: str, *, scope: BackendAccess) -> StoredNodeSummaries:
            # First node read: commit a text-change update (new text, bumped
            # version, replaced node rows, same-second updated_at) between
            # navigate's text read and its node read, then serve the NEW rows.
            nonlocal update_committed
            if not update_committed:
                update_committed = True

                def _commit_update(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        'UPDATE context_entries SET text_content = ?, '
                        'version = version + 1, updated_at = CURRENT_TIMESTAMP '
                        'WHERE id = ?',
                        ('# New\nnew body\n', context_id),
                    )

                await nav_backend.execute_write(_commit_update, access_scope=MAINTENANCE)
                await repos.index_nodes.replace_nodes_for_context(context_id, [
                    IndexNodeRow(
                        node_id='new', level=1, ordinal=1, title='New',
                        node_summary='New section abstract.', char_start=0, char_end=15,
                    ),
                ], scope=MAINTENANCE)
            return await real_get_nodes(context_id, scope=scope)

        with patch.object(repos.index_nodes, 'get_nodes_for_context', racing_get_nodes):
            result = await navigate_as_dict(context_id=cid, include_node_summaries=True)

        first = result['root']['children'][0]
        assert first['title'] == 'New'
        assert first['summary'] == 'New section abstract.'

    @pytest.mark.asyncio
    async def test_navigate_unshare_inside_the_window_reports_not_found(self, nav_backend: StorageBackend) -> None:
        """An unshare revoking the caller's read access between the two reads ends in not found.

        The default principal reads alice's private entry through a read grant. Alice
        removes that grant after navigate read the text and before it read the node
        summaries: the closing probe no longer sees the entry, the snapshot is retaken,
        and the retaken read finds nothing, so neither the outline nor the node
        summaries read inside the window reach the caller.
        """
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await store_entry(nav_backend, '# Alice\nshared body\n', owner='alice')
        await insert_grant(nav_backend, cid, 'user', 'local', 'read', 'alice')
        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='alice', level=1, ordinal=1, title='Alice',
                node_summary='Shared section abstract.', char_start=0, char_end=20,
            ),
        ], scope=MAINTENANCE)

        real_get_nodes = repos.index_nodes.get_nodes_for_context
        revoked = False

        async def revoking_get_nodes(context_id: str, *, scope: BackendAccess) -> StoredNodeSummaries:
            nonlocal revoked
            if not revoked:
                revoked = True
                with as_principal('alice'):
                    await unshare_context(
                        context_id=context_id, principal_type='user', principal_id='local', permission='read',
                    )
            return await real_get_nodes(context_id, scope=scope)

        with (
            patch.object(repos.index_nodes, 'get_nodes_for_context', revoking_get_nodes),
            pytest.raises(ToolError) as error,
        ):
            await navigate_as_dict(context_id=cid, include_node_summaries=True)

        assert revoked
        assert str(error.value) == f'Context entry not found: {cid}'

    @pytest.mark.asyncio
    async def test_summaries_omitted_by_default(self, nav_backend: StorageBackend) -> None:
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await store_entry(nav_backend, '# Intro\nhello\n## Details\nmore body here\n')
        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='intro/details', level=2, ordinal=1, title='Details',
                node_summary='Covers the details.', char_start=0, char_end=10,
            ),
        ], scope=MAINTENANCE)
        # Default include_node_summaries=False: descendant summaries stay None.
        result = await navigate_as_dict(context_id=cid)
        assert result['root']['children'][0]['children'][0]['summary'] is None

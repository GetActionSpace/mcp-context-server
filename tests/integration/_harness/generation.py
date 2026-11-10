"""Real-server checks of deferred generation.

A write without ``defer_generation`` generates its embeddings and summary before it
returns and reports ``generation_status: 'ready'``. A deferred write stores the entry at
once and reports ``'pending'``: the keyword and full-text searches find the entry straight
away, semantic search finds it only once the server's generation worker has embedded it,
and from then on it reports ``'ready'``. A job whose generation keeps failing ends
``'failed'``, its entry still readable and full-text searchable, and the next server start
retries it.

The checks on the primary server run against its real embedding provider. The checks
that must hold a job pending, or make it fail, start their own servers on an isolated
database (``_isolated_database``), so no other server's worker claims their jobs, and
embed through an endpoint the test process serves (:mod:`tests.integration._ollama_stub`):
one that holds the embedding of a marked text until the check releases it, or one that
refuses every document embedding. The failure check then starts a server on the same
database with the run's real provider, which retries the failed job when it starts.
Another pair of servers shares one isolated database, the owner's and ``bob``'s: ``bob``
reads the generation state of the owner's public deferred entry, while the owner's
private pending entries stay absent from his reads and from his statistics.
"""

import asyncio
import time
from typing import Any

from fastmcp import Client

from tests.integration._harness.access_scoping import SECOND_PRINCIPAL
from tests.integration._harness.core import HarnessCore
from tests.integration._ollama_stub import OllamaStub
from tests.integration._ollama_stub import serve_ollama_embeddings
from tests.integration._ollama_stub import stub_generation_env

READY = 'ready'
PENDING = 'pending'
FAILED = 'failed'
# How long a check waits for the worker to finish a job with the real providers: a
# summary of a few hundred words takes tens of seconds on a CPU.
GENERATION_TIMEOUT_S = 120.0
# How long a check waits for a job whose every embedding the stub refuses to end failed,
# or for a job the stub embeds at once to finish.
STUB_TIMEOUT_S = 60.0
GENERATION_POLL_S = 0.5
# A word the holding stub recognizes: an embedding of a text carrying it waits for the release.
HOLD_MARKER = 'heldquartzmarker'
# The vector length of the stub servers that share no database with a real provider.
STUB_DIMENSION = 64
# Embeddings are the only output a stub server generates, so a job's outcome is its embedding's.
NO_SUMMARIES = {'ENABLE_SUMMARY_GENERATION': 'false', 'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'false'}
# One provider attempt per batch and one attempt per job, so a refused embedding fails the job at once.
SINGLE_ATTEMPT = {'EMBEDDING_RETRY_MAX_ATTEMPTS': '1', 'GENERATION_JOB_MAX_ATTEMPTS': '1'}

SYNC_TEXT = 'Tidal turbines turn the movement of ocean currents into electricity for coastal towns.'
SYNC_QUERY = 'ocean current power generation'
DEFERRED_TOKEN = 'lumenkeel'
# Longer than the default SUMMARY_MIN_CONTENT_LENGTH, so the job generates a summary too.
DEFERRED_TEXT = (
    'Lighthouse keepers once trimmed wicks every night, polished the brass and wound the clockwork that '
    f'turned the lens. The {DEFERRED_TOKEN} logbook records each watch: the weather at dusk, the hour the lamp '
    'was lit, the ships that passed and any fog that called for the horn. A Fresnel lens gathers the light of '
    'a small flame into a beam that sailors see more than twenty nautical miles away, because its concentric '
    'rings bend the light the way a much thicker lens would. Automation replaced most keepers during the '
    'twentieth century, but the old towers still guide boats along rocky coasts, and volunteers keep the '
    'logbooks, the lenses and the stories of the families who lived beside them. Each tower flashes its own '
    'pattern, so a navigator who counts the flashes knows which headland lies ahead on a moonless night.'
)
DEFERRED_QUERY = 'lighthouse lens beam guiding ships at night'
OLD_TOKEN = 'basalt'
OLD_TEXT = (
    'Basalt columns form when a thick lava flow cools slowly and contracts, cracking into tall hexagonal '
    'pillars like those along the coasts of Iceland and Northern Ireland.'
)
NEW_TOKEN = 'sourdough'
NEW_TEXT = (
    'A sourdough starter is a culture of wild yeast and lactic acid bacteria that ferments flour and water, '
    'giving the bread its sour flavor and its open crumb.'
)
FAILING_TOKEN = 'pearblossom'
FAILING_TEXT = f'Orchard notes: the {FAILING_TOKEN} came a week early this spring.'
BATCH_FLAG_ERROR = 'defer_generation must be a boolean'
PRIVATE_GENERATION_TOKEN = 'cinderwhorl'
PUBLIC_GENERATION_TOKEN = 'mosslantern'


def stub_server_env(stub: OllamaStub) -> dict[str, str]:
    """Return the overrides of a server whose only generated output is its embeddings through the stub.

    Args:
        stub: The running stub.

    Returns:
        The overrides.
    """
    return {**stub_generation_env(stub), **NO_SUMMARIES}


async def tool_content(client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool that must succeed and return its structured content.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The structured content.

    Raises:
        AssertionError: If the tool refused the call.
    """
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    if result.is_error or not isinstance(result.structured_content, dict):
        raise AssertionError(f'{tool} failed: {result.content}')
    return result.structured_content


async def entries_by_id(client: Client[Any], context_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return the entries a principal reads through ``get_context_by_ids``, keyed by id.

    Args:
        client: The reading principal's client.
        context_ids: The ids to fetch.

    Returns:
        The returned entries.
    """
    rows = (await tool_content(client, 'get_context_by_ids', {'context_ids': context_ids}))['result']
    return {str(row['id']): row for row in rows}


async def generation_states(client: Client[Any], context_ids: list[str]) -> dict[str, str]:
    """Return the generation state of each entry a principal reads.

    Args:
        client: The reading principal's client.
        context_ids: The entries.

    Returns:
        Each readable entry's ``generation_status`` by id.
    """
    entries = await entries_by_id(client, context_ids)
    return {context_id: str(row['generation_status']) for context_id, row in entries.items()}


async def await_states(client: Client[Any], context_ids: list[str], state: str, timeout_s: float) -> dict[str, str]:
    """Poll the entries until every one reports ``state`` or the time runs out.

    Args:
        client: The reading principal's client.
        context_ids: The entries.
        state: The generation state to wait for.
        timeout_s: The longest wait.

    Returns:
        The states read last.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        states = await generation_states(client, context_ids)
        if states == dict.fromkeys(context_ids, state) or time.monotonic() >= deadline:
            return states
        await asyncio.sleep(GENERATION_POLL_S)


async def generation_counts(client: Client[Any]) -> dict[str, int]:
    """Return the ``generation`` block of the statistics a principal is shown.

    Args:
        client: The principal's client.

    Returns:
        The pending and failed counts.
    """
    block = (await tool_content(client, 'get_statistics', {}))['generation']
    return {'pending': int(block['pending']), 'failed': int(block['failed'])}


async def matched_ids(client: Client[Any], tool: str, arguments: dict[str, Any]) -> list[str]:
    """Run a search tool and return the ids of its results in order.

    Args:
        client: The searching principal's client.
        tool: A search tool.
        arguments: The tool arguments.

    Returns:
        The returned entry ids.
    """
    return [str(row['id']) for row in (await tool_content(client, tool, arguments))['results']]


async def semantic_distance(client: Client[Any], query: str, thread_id: str, context_id: str) -> float | None:
    """Return the semantic distance of one entry to a query, or None when semantic search does not return it.

    Args:
        client: The searching principal's client.
        query: The query text.
        thread_id: The thread to search, holding the entry alone.
        context_id: The entry.

    Returns:
        The entry's ``semantic_distance``, lower meaning closer.
    """
    results = (await tool_content(client, 'semantic_search_context', {'query': query, 'thread_id': thread_id}))['results']
    for row in results:
        if str(row['id']) == context_id:
            return float(row['scores']['semantic_distance'])
    return None


class GenerationMixin(HarnessCore):
    """Checks of synchronous and deferred generation, the failed state and its recovery."""

    def _skip(self, test_name: str, reason: str) -> bool:
        """Record a check that does not apply to this run.

        Args:
            test_name: The check's name.
            reason: Why it does not apply.

        Returns:
            bool: Always True.
        """
        self.test_results.append((test_name, True, f'Skipped ({reason})'))
        return True

    async def test_generation_synchronous_store_ready(self) -> bool:
        """A store without ``defer_generation`` reports ready and semantic search finds it at once.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_synchronous_store_ready'
        assert self.client is not None
        if 'semantic_search_context' not in self.registered_tools:
            return self._skip(test_name, 'semantic_search_context is not registered')
        try:
            thread = f'{self.test_thread_id}_generation_sync'
            arguments = {'thread_id': thread, 'source': 'agent', 'text': SYNC_TEXT}
            stored = await tool_content(self.client, 'store_context', arguments)
            context_id = str(stored['context_id'])
            problems: list[str] = []
            if stored.get('generation_status') != READY:
                problems.append(f'the store reported {stored.get("generation_status")!r}, expected {READY!r}')
            if (states := await generation_states(self.client, [context_id])) != {context_id: READY}:
                problems.append(f'get_context_by_ids reported {states}')
            if await semantic_distance(self.client, SYNC_QUERY, thread, context_id) is None:
                problems.append('semantic search did not find the entry right after the store')
            return self._record(test_name, problems, 'ready at once and found by semantic search')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_generation_deferred_store_turns_ready(self) -> bool:
        """A deferred store is pending, searchable by keyword and full text at once, and semantic only once ready.

        Semantic search runs right after the store, and the entry's state is read after
        it: an entry still pending then was pending during the search, so it must have
        been absent. The worker may finish first on a fast provider; the detail says so.
        Once the entry reports ready, semantic search finds it and, when the server
        generates summaries, its summary is present.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_deferred_store_turns_ready'
        assert self.client is not None
        if not {'semantic_search_context', 'fts_search_context'}.issubset(self.registered_tools):
            return self._skip(test_name, 'semantic_search_context or fts_search_context is not registered')
        try:
            summary = (await tool_content(self.client, 'get_statistics', {}))['summary']
            expects_summary = bool(summary.get('enabled') and summary.get('available')) and len(DEFERRED_TEXT) >= int(
                summary.get('min_content_length', 0),
            )
            thread = f'{self.test_thread_id}_generation_deferred'
            stored = await tool_content(self.client, 'store_context', {
                'thread_id': thread, 'source': 'agent', 'text': DEFERRED_TEXT, 'defer_generation': True,
            })
            context_id = str(stored['context_id'])
            problems: list[str] = []
            if stored.get('generation_status') != PENDING:
                problems.append(f'the deferred store reported {stored.get("generation_status")!r}, expected {PENDING!r}')
            fts = {'query': DEFERRED_TOKEN, 'thread_id': thread}
            if context_id not in await matched_ids(self.client, 'fts_search_context', fts):
                problems.append('fts_search_context did not find the entry right after the store')
            if context_id not in await matched_ids(self.client, 'search_context', {'thread_id': thread}):
                problems.append('search_context did not find the entry right after the store')
            distance = await semantic_distance(self.client, DEFERRED_QUERY, thread, context_id)
            state_after_search = (await generation_states(self.client, [context_id])).get(context_id)
            if state_after_search == PENDING and distance is not None:
                problems.append('semantic search returned the entry while its generation was pending')
            observed = 'absent from semantic search while pending' if state_after_search == PENDING else (
                f'reported {state_after_search!r} before the pending read could run'
            )

            started = time.monotonic()
            if (states := await await_states(self.client, [context_id], READY, GENERATION_TIMEOUT_S)) != {context_id: READY}:
                problems.append(f'generation did not finish within {GENERATION_TIMEOUT_S:.0f} s: {states}')
                return self._record(test_name, problems, '')
            elapsed = time.monotonic() - started
            if await semantic_distance(self.client, DEFERRED_QUERY, thread, context_id) is None:
                problems.append('semantic search did not find the entry once it was ready')
            if expects_summary:
                rows = (await tool_content(self.client, 'search_context', {'thread_id': thread}))['results']
                if not any(str(row['id']) == context_id and row.get('summary') for row in rows):
                    problems.append('the ready entry has no summary although the server generates summaries')
            return self._record(
                test_name, problems,
                f'pending, found by FTS and keyword search at once, {observed}, ready {elapsed:.1f} s later, '
                f'then found by semantic search{" with its summary" if expects_summary else ""}',
            )
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_generation_deferred_update_replaces_vectors(self) -> bool:
        """A deferred text update drops the old vectors at once and the worker embeds the new text.

        The entry holds its thread alone. Before the update, the old text is closer to
        the entry's vectors than the new text; right after the deferred update, the
        old vectors are gone, so semantic search either misses the entry or measures it
        well away from the old text; once ready, the new text is the closer one.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_deferred_update_replaces_vectors'
        assert self.client is not None
        if not {'semantic_search_context', 'fts_search_context'}.issubset(self.registered_tools):
            return self._skip(test_name, 'semantic_search_context or fts_search_context is not registered')
        try:
            thread = f'{self.test_thread_id}_generation_update'
            arguments = {'thread_id': thread, 'source': 'agent', 'text': OLD_TEXT}
            stored = await tool_content(self.client, 'store_context', arguments)
            context_id = str(stored['context_id'])
            old_before = await semantic_distance(self.client, OLD_TEXT, thread, context_id)
            new_before = await semantic_distance(self.client, NEW_TEXT, thread, context_id)
            if old_before is None or new_before is None or not old_before < new_before:
                apart = f'the stored entry does not tell the texts apart: old {old_before}, new {new_before}'
                return self._record(test_name, [apart], '')
            midpoint = (old_before + new_before) / 2

            updated = await tool_content(self.client, 'update_context', {
                'context_id': context_id, 'text': NEW_TEXT, 'defer_generation': True,
            })
            problems: list[str] = []
            if updated.get('generation_status') != PENDING:
                problems.append(f'the deferred update reported {updated.get("generation_status")!r}, expected {PENDING!r}')
            entry = (await entries_by_id(self.client, [context_id])).get(context_id, {})
            if entry.get('text_content') != NEW_TEXT:
                problems.append('get_context_by_ids did not return the new text right after the update')
            new_fts = {'query': NEW_TOKEN, 'thread_id': thread}
            if context_id not in await matched_ids(self.client, 'fts_search_context', new_fts):
                problems.append('fts_search_context did not find the new text right after the update')
            if context_id in await matched_ids(self.client, 'fts_search_context', {'query': OLD_TOKEN, 'thread_id': thread}):
                problems.append('fts_search_context still found the old text after the update')
            old_right_after = await semantic_distance(self.client, OLD_TEXT, thread, context_id)
            if old_right_after is not None and old_right_after <= midpoint:
                problems.append(f'semantic search still measured the old vectors after the update: {old_right_after}')

            if (states := await await_states(self.client, [context_id], READY, GENERATION_TIMEOUT_S)) != {context_id: READY}:
                problems.append(f'generation did not finish within {GENERATION_TIMEOUT_S:.0f} s: {states}')
                return self._record(test_name, problems, '')
            new_after = await semantic_distance(self.client, NEW_TEXT, thread, context_id)
            old_after = await semantic_distance(self.client, OLD_TEXT, thread, context_id)
            if new_after is None or old_after is None or not new_after < old_after:
                problems.append(f'the ready entry is not closer to the new text: new {new_after}, old {old_after}')
            return self._record(test_name, problems, 'old vectors dropped by the update, new text embedded by the worker')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_generation_deferred_batch_counts(self) -> bool:
        """A batch mixes deferred and synchronous entries, refuses an unreadable flag, and counts what is pending.

        The batch runs on a server whose stub holds every embedding of a text carrying
        :data:`HOLD_MARKER`, so the deferred entries stay pending until the check
        releases the stub, and the counts read while they wait are exact.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_deferred_batch_counts'
        if 'store_context_batch' not in self.registered_tools:
            return self._skip(test_name, 'store_context_batch is not registered')
        try:
            thread = f'{self.test_thread_id}_generation_batch'
            flags: list[tuple[str, dict[str, Any], str | None]] = [
                (f'Held batch entry one {HOLD_MARKER}.', {'defer_generation': True}, PENDING),
                ('Batch entry embedded in the request.', {'defer_generation': 'false'}, READY),
                (f'Held batch entry two {HOLD_MARKER}.', {'defer_generation': 'on'}, PENDING),
                ('Batch entry with an unreadable flag.', {'defer_generation': 'maybe'}, None),
                ('Batch entry without the flag.', {}, READY),
            ]
            entries = [{'thread_id': thread, 'source': 'agent', 'text': text, **flag} for text, flag, _ in flags]
            with serve_ollama_embeddings(STUB_DIMENSION, hold_marker=HOLD_MARKER) as stub:
                async with self._isolated_database() as database, self._second_server(
                    stub_server_env(stub), isolated_database=database,
                ) as server:
                    batch = await tool_content(server, 'store_context_batch', {'entries': entries, 'atomic': False})
                    problems = self._batch_result_problems(batch['results'], [expected for _, _, expected in flags])
                    held = [str(row['context_id']) for row in batch['results'] if row.get('generation_status') == PENDING]
                    if (counts := await generation_counts(server)) != {'pending': len(held), 'failed': 0}:
                        problems.append(f'get_statistics counted {counts} while {len(held)} entries were held')
                    stub.release()
                    if (states := await await_states(server, held, READY, STUB_TIMEOUT_S)) != dict.fromkeys(held, READY):
                        problems.append(f'the released entries did not turn ready: {states}')
                    if (counts := await generation_counts(server)) != {'pending': 0, 'failed': 0}:
                        problems.append(f'get_statistics counted {counts} once every job finished')
            return self._record(test_name, problems, 'per-entry states, the flag refusal and the pending count all exact')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    @staticmethod
    def _batch_result_problems(results: list[dict[str, Any]], expected: list[str | None]) -> list[str]:
        """Compare each batch result with the state its flag asks for.

        Args:
            results: The batch's per-entry results.
            expected: Per entry, the generation state it reports, or None for an entry
                the flag check refuses.

        Returns:
            The problems found.
        """
        problems: list[str] = []
        by_index = {int(row['index']): row for row in results}
        for index, state in enumerate(expected):
            row = by_index.get(index, {})
            if state is None:
                refused = not row.get('success') and BATCH_FLAG_ERROR in str(row.get('error'))
                if not refused or row.get('generation_status') is not None:
                    problems.append(f'entry {index} with an unreadable flag gave {row}')
            elif not row.get('success') or row.get('generation_status') != state:
                problems.append(f'entry {index} gave {row}, expected {state!r}')
        return problems

    async def test_generation_failed_job_recovers_on_restart(self) -> bool:
        """A job whose embedding keeps failing ends failed, and a restart with a working provider retries it.

        Server A embeds through a stub that refuses every document embedding, with one
        attempt per provider call and per job: the deferred entry turns failed, stays
        readable and full-text searchable, and the statistics count it. Server B then
        starts on the same database with the run's real provider, the model and the
        dimension of the primary server; its worker resets the failed job when it
        starts, and the entry turns ready and semantic search finds it.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_failed_job_recovers_on_restart'
        assert self.client is not None
        if not {'semantic_search_context', 'fts_search_context'}.issubset(self.registered_tools):
            return self._skip(test_name, 'semantic_search_context or fts_search_context is not registered')
        try:
            semantic = (await tool_content(self.client, 'get_statistics', {}))['semantic_search']
            model, dimension = str(semantic['model']), int(semantic['dimensions'])
            real_provider = {'EMBEDDING_MODEL': model, 'EMBEDDING_DIM': str(dimension), **NO_SUMMARIES}
            thread = f'{self.test_thread_id}_generation_failed'
            problems: list[str] = []
            with serve_ollama_embeddings(dimension, refuse_embeddings=True) as stub:
                refusing = {**stub_server_env(stub), 'EMBEDDING_MODEL': model, **SINGLE_ATTEMPT}
                async with self._isolated_database() as database:
                    async with self._second_server(refusing, isolated_database=database) as server_a:
                        context_id = await self._failed_entry(server_a, thread, problems)
                    async with self._second_server(real_provider, isolated_database=database) as server_b:
                        await self._recovery_problems(server_b, thread, context_id, problems)
            return self._record(test_name, problems, 'failed under a refusing provider, ready after a restart')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def _failed_entry(self, server: Client[Any], thread: str, problems: list[str]) -> str:
        """Store a deferred entry on the refusing server and check its failed state.

        Args:
            server: The client of the server whose provider refuses every document embedding.
            thread: The thread to store into.
            problems: The list the problems found are appended to.

        Returns:
            The entry's id.
        """
        stored = await tool_content(server, 'store_context', {
            'thread_id': thread, 'source': 'agent', 'text': FAILING_TEXT, 'defer_generation': True,
        })
        context_id = str(stored['context_id'])
        if stored.get('generation_status') != PENDING:
            problems.append(f'the deferred store reported {stored.get("generation_status")!r}, expected {PENDING!r}')
        if (states := await await_states(server, [context_id], FAILED, STUB_TIMEOUT_S)) != {context_id: FAILED}:
            problems.append(f'the refused job did not end failed within {STUB_TIMEOUT_S:.0f} s: {states}')
        entry = (await entries_by_id(server, [context_id])).get(context_id, {})
        if entry.get('text_content') != FAILING_TEXT:
            problems.append('the failed entry is not readable through get_context_by_ids')
        if context_id not in await matched_ids(server, 'fts_search_context', {'query': FAILING_TOKEN, 'thread_id': thread}):
            problems.append('fts_search_context did not find the failed entry')
        if (counts := await generation_counts(server)) != {'pending': 0, 'failed': 1}:
            problems.append(f'get_statistics counted {counts} with one failed job')
        return context_id

    async def _recovery_problems(self, server: Client[Any], thread: str, context_id: str, problems: list[str]) -> None:
        """Check that the restarted server's worker retries the failed job to completion.

        Args:
            server: The client of the server with the real provider.
            thread: The entry's thread.
            context_id: The failed entry.
            problems: The list the problems found are appended to.
        """
        if (states := await await_states(server, [context_id], READY, GENERATION_TIMEOUT_S)) != {context_id: READY}:
            problems.append(f'the restarted server left the job unfinished after {GENERATION_TIMEOUT_S:.0f} s: {states}')
            return
        if await semantic_distance(server, FAILING_TEXT, thread, context_id) is None:
            problems.append('semantic search did not find the recovered entry')
        if (counts := await generation_counts(server)) != {'pending': 0, 'failed': 0}:
            problems.append(f'get_statistics counted {counts} after the recovery')

    async def test_generation_second_principal_reads_public_state(self) -> bool:
        """Show the second principal the generation state of a public entry, and nothing of the private ones.

        Two servers serve one isolated database, the owner's under the default principal
        and ``bob``'s, both embedding through a stub that holds every embedding of a text
        carrying :data:`HOLD_MARKER`, so the owner's deferred entries stay pending until
        the check releases the stub. While they wait, ``bob`` reads the public entry as
        pending, finds none of the private ones through any read, and his statistics
        count the public entry alone; once released, he reads it as ready and his counts
        return to zero.

        Returns:
            bool: True if the check passed or does not apply.
        """
        test_name = 'generation_second_principal_reads_public_state'
        if not {'store_context_batch', 'fts_search_context'}.issubset(self.registered_tools):
            return self._skip(test_name, 'store_context_batch or fts_search_context is not registered')
        try:
            with serve_ollama_embeddings(STUB_DIMENSION, hold_marker=HOLD_MARKER) as stub:
                env = stub_server_env(stub)
                async with self._isolated_database() as database, self._second_server(
                    env, isolated_database=database,
                ) as owner, self._second_server(
                    {**env, 'ACCESS_CONTROL_DEFAULT_PRINCIPAL': SECOND_PRINCIPAL}, isolated_database=database,
                ) as bob:
                    problems = await self._held_generation_problems(owner, bob, stub)
            return self._record(test_name, problems, f"{SECOND_PRINCIPAL} sees the public entry's state alone")
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def _held_generation_problems(self, owner: Client[Any], bob: Client[Any], stub: OllamaStub) -> list[str]:
        """Store the owner's held entries, compare what each principal reads, then release them.

        Args:
            owner: The client of the server running as the default principal.
            bob: The client of the server running as the second principal.
            stub: The holding stub both servers embed through.

        Returns:
            The problems found.
        """
        private_thread = f'{self.test_thread_id}_generation_private'
        public_thread = f'{self.test_thread_id}_generation_public'
        entries = [
            {'thread_id': private_thread, 'source': 'agent', 'visibility': 'private', 'defer_generation': True,
             'text': f'Private held entry {number} carrying {PRIVATE_GENERATION_TOKEN} {HOLD_MARKER}.'}
            for number in (1, 2)
        ]
        entries.append({'thread_id': public_thread, 'source': 'agent', 'visibility': 'public', 'defer_generation': True,
                        'text': f'Public held entry carrying {PUBLIC_GENERATION_TOKEN} {HOLD_MARKER}.'})
        batch = await tool_content(owner, 'store_context_batch', {'entries': entries})
        ids = [str(row['context_id']) for row in batch['results']]
        public_id = ids[-1]
        problems: list[str] = []
        if [row.get('generation_status') for row in batch['results']] != [PENDING] * len(ids):
            problems.append(f'the deferred batch reported {batch["results"]}')

        if (states := await generation_states(bob, ids)) != {public_id: PENDING}:
            problems.append(f'{SECOND_PRINCIPAL} read the states {states} while every job was held')
        for tool, arguments, expected in (
            ('search_context', {'thread_id': private_thread}, []),
            ('search_context', {'thread_id': public_thread}, [public_id]),
            ('fts_search_context', {'query': PRIVATE_GENERATION_TOKEN}, []),
            ('fts_search_context', {'query': PUBLIC_GENERATION_TOKEN}, [public_id]),
        ):
            if (found := await matched_ids(bob, tool, arguments)) != expected:
                problems.append(f'{tool} {arguments} by {SECOND_PRINCIPAL} returned {found}, expected {expected}')
        for who, client, expected_pending in (('owner', owner, len(ids)), (SECOND_PRINCIPAL, bob, 1)):
            if (counts := await generation_counts(client)) != {'pending': expected_pending, 'failed': 0}:
                problems.append(f'the {who} counted {counts} while every job was held')

        stub.release()
        if (states := await await_states(owner, ids, READY, STUB_TIMEOUT_S)) != dict.fromkeys(ids, READY):
            problems.append(f'the released jobs did not finish: {states}')
        if (states := await generation_states(bob, ids)) != {public_id: READY}:
            problems.append(f'{SECOND_PRINCIPAL} read the states {states} once every job finished')
        if (counts := await generation_counts(bob)) != {'pending': 0, 'failed': 0}:
            problems.append(f'{SECOND_PRINCIPAL} counted {counts} once every job finished')
        return problems

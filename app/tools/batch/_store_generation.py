"""The generation pass of the store_context_batch tool.

Every output an entry stores is generated here, before any transaction opens: the
embeddings and the summary of each entry (reusing a likely duplicate's), the node
summaries of the entries that survive the pass, and the outputs of an entry whose
deduplication diverged inside its transaction. ``StoreGeneration`` holds the outputs
by the entry's position among the validated entries, with the response counters they
feed.
"""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.errors import format_exception_message
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._generation import generate_compression_with_timeout
from app.tools._generation import generate_embeddings_with_timeout
from app.tools._generation import generate_index_nodes_with_timeout
from app.tools._generation import generate_summary_with_timeout
from app.tools._generation import generation_active

if TYPE_CHECKING:
    from app.embeddings import EmbeddingProvider
    from app.repositories import RepositoryContainer
    from app.summary import SummaryProvider

logger = logging.getLogger(__name__)


@dataclass
class StoreGeneration:
    """The generated outputs of a batch store's entries and the response counters they feed.

    Every map is keyed by the entry's position among the validated entries.

    Attributes:
        deferred: The entries whose generation is left to a job.
        embeddings: Each entry's compressed chunk embeddings; None when none were generated.
        summaries: Each entry's summary, generated or reused from its likely duplicate.
        likely_duplicates: The id of each entry's likely duplicate found by the pre-check.
        preserved_summaries: The entries whose summary was reused from the likely duplicate.
        index_nodes: Each surviving entry's node summaries; None leaves stored rows untouched.
        errors: ``(index, error)`` of each entry whose generation failed in a non-atomic batch.
        embeddings_generated: Entries holding generated embeddings.
        summaries_generated: Entries holding a summary a model call produced.
        summaries_preserved: Entries holding a reused summary.
    """

    deferred: set[int]
    embeddings: dict[int, list[ChunkEmbedding] | None] = field(default_factory=dict[int, list[ChunkEmbedding] | None])
    summaries: dict[int, str | None] = field(default_factory=dict[int, str | None])
    likely_duplicates: dict[int, str | None] = field(default_factory=dict[int, str | None])
    preserved_summaries: set[int] = field(default_factory=set[int])
    index_nodes: dict[int, list[IndexNodeRow] | None] = field(default_factory=dict[int, list[IndexNodeRow] | None])
    errors: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    embeddings_generated: int = 0
    summaries_generated: int = 0
    summaries_preserved: int = 0

    def discard_counts(self, ve_idx: int) -> None:
        """Reverse a discarded entry's response-counter contributions.

        A transaction-phase failure discards an entry AFTER the generation
        phase bumped the response counters. Without compensation the
        response message computes not_stored = generated - stored and
        labels the whole gap "not stored - duplicates", so a FAILED
        never-stored entry would be reported as a dedup skip, and
        "summaries generated/preserved" would be claimed for entries that
        stored nothing. The decrement criteria mirror the bump criteria
        exactly: one embeddings bump iff the entry holds compressed
        embeddings, one preserved bump iff its index is still in the
        preserved set, else one generated-summary bump iff it holds a
        non-empty generated summary string.

        Args:
            ve_idx: The entry's position among the validated entries.
        """
        if self.embeddings.get(ve_idx) is not None:
            self.embeddings_generated -= 1
        if ve_idx in self.preserved_summaries:
            self.preserved_summaries.discard(ve_idx)
            self.summaries_preserved -= 1
        else:
            summary_value = self.summaries.get(ve_idx)
            if isinstance(summary_value, str) and summary_value:
                self.summaries_generated -= 1


async def generate_store_outputs(
    repos: 'RepositoryContainer',
    validated_entries: list[dict[str, Any]],
    *,
    scope: AccessScope,
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
    min_content_length: int,
    atomic: bool,
) -> StoreGeneration:
    """Generate each validated entry's embeddings and summary, outside any transaction.

    Entries are processed sequentially to limit provider load; each entry runs its
    embedding and summary legs in parallel. An entry that deduplicates first asks the
    read-only pre-check for a likely duplicate, whose stored embeddings and summary it
    reuses. An entry asking to defer while a generation provider is active generates
    nothing.

    Args:
        repos: Repository container.
        validated_entries: The entries that passed validation.
        scope: The caller's scope, which the pre-check reads under.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.
        min_content_length: The shortest text that gets a summary; 0 admits every text.
        atomic: Whether the batch is all-or-nothing.

    Returns:
        The generated outputs, with ``errors`` naming each entry whose generation failed.

    Raises:
        ToolError: In atomic mode, on the first entry whose generation or compression failed.
    """
    # The entries whose generation is left to a job: those asking for it while a
    # generation provider is active. Without one there is nothing to defer, and such
    # an entry is stored like any other.
    generation_can_defer = generation_active()
    generation = StoreGeneration(deferred={
        ve_idx for ve_idx, entry in enumerate(validated_entries)
        if entry['defer_generation'] and generation_can_defer
    })

    for ve_idx, entry in enumerate(validated_entries):
        original_idx = entry['index']
        text_content = entry['text_content']
        if ve_idx in generation.deferred:
            generation.embeddings[ve_idx] = None
            generation.summaries[ve_idx] = None
            generation.likely_duplicates[ve_idx] = None
            continue

        tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
        task_names: list[str] = []

        # Performance optimization: pre-check for likely duplicates (read-only).
        # The candidate id and its stored summary come from ONE
        # statement-level snapshot (see DuplicateCandidate): a separate
        # later summary read could observe a row version a concurrent
        # update committed in between, pairing a reused summary with text
        # it does not describe. An entry that does not deduplicate always
        # inserts, so it skips the pre-check and every leg runs for it.
        likely_duplicate_id: str | None = None
        duplicate_summary: str | None = None
        if entry['deduplicate'] and (embedding_provider is not None or summary_provider is not None):
            duplicate_candidate = await repos.context.check_latest_is_duplicate(
                thread_id=entry['thread_id'],
                source=entry['source'],
                text_content=text_content,
                scope=scope,
            )
            if duplicate_candidate is not None:
                likely_duplicate_id = duplicate_candidate.context_id
                duplicate_summary = duplicate_candidate.summary
        generation.likely_duplicates[ve_idx] = likely_duplicate_id

        # Embedding task (with pre-check optimization)
        if embedding_provider is not None:
            if likely_duplicate_id is not None:
                has_embeddings = await repos.embeddings.exists(likely_duplicate_id, scope=scope)
                if has_embeddings:
                    logger.debug(
                        'Pre-check: skipping embedding generation for likely duplicate '
                        'of context %s at index %d',
                        likely_duplicate_id, original_idx,
                    )
                else:
                    tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                    task_names.append('embedding')
            else:
                tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                task_names.append('embedding')

        # Summary task (with pre-check optimization)
        if summary_provider is not None:
            if min_content_length > 0 and len(text_content) < min_content_length:
                logger.info(
                    'Skipping summary generation at index %d: '
                    'text length %d < min_content_length %d',
                    original_idx, len(text_content), min_content_length,
                )
                generation.summaries[ve_idx] = None
            elif likely_duplicate_id is not None:
                if duplicate_summary is not None:
                    generation.summaries[ve_idx] = duplicate_summary
                    generation.summaries_preserved += 1
                    generation.preserved_summaries.add(ve_idx)
                    logger.debug(
                        'Pre-check: reusing existing summary for likely duplicate '
                        'of context %s at index %d',
                        likely_duplicate_id, original_idx,
                    )
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                    task_names.append('summary')
            else:
                tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                task_names.append('summary')

        if not tasks_to_run:
            generation.embeddings[ve_idx] = None
            if ve_idx not in generation.summaries:
                generation.summaries[ve_idx] = None
            continue

        gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        errors: list[tuple[str, BaseException]] = []
        for name, result in zip(task_names, gather_results, strict=True):
            if isinstance(result, BaseException):
                errors.append((name, result))
                logger.error(
                    'Generation failed for %s at index %d (retries exhausted): %s',
                    name, original_idx, result,
                )
            elif name == 'embedding':
                generation.embeddings[ve_idx] = cast(list[ChunkEmbedding] | None, result)
            elif name == 'summary':
                generation.summaries[ve_idx] = cast(str | None, result)

        if errors:
            error_details = '; '.join(
                f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
            )
            if atomic:
                raise ToolError(
                    f'Generation failed at index {original_idx} after exhausting '
                    f'configured retries: {error_details}. No data was saved.',
                )
            generation.errors.append((original_idx, f'Generation failed: {error_details}'))
            generation.embeddings.setdefault(ve_idx, None)
            generation.summaries.setdefault(ve_idx, None)
            # This entry is discarded by the caller; undo the preserved-summary count
            # it provisionally bumped in the pre-check, mirroring the
            # compression-failure compensation -- counts must reflect entries
            # surviving the generation phase.
            if ve_idx in generation.preserved_summaries:
                generation.preserved_summaries.discard(ve_idx)
                generation.summaries_preserved -= 1
        else:
            generation.embeddings.setdefault(ve_idx, None)
            if ve_idx not in generation.summaries:
                generation.summaries[ve_idx] = None

            # Compress embeddings OUTSIDE the DB transaction (mirrors
            # the generation-first invariant). No-op when
            # ENABLE_EMBEDDING_COMPRESSION=false. In atomic mode a
            # failure aborts the whole batch; in non-atomic mode this
            # entry is marked failed and processing continues.
            try:
                generation.embeddings[ve_idx] = await generate_compression_with_timeout(
                    generation.embeddings.get(ve_idx),
                )
            except ToolError as compress_err:
                if atomic:
                    raise ToolError(
                        f'Compression failed at index {original_idx}: '
                        f'{compress_err}. No data was saved.',
                    ) from compress_err
                generation.errors.append(
                    (original_idx, f'Compression failed: {compress_err}'),
                )
                generation.embeddings[ve_idx] = None
                generation.summaries[ve_idx] = None
                # This entry is discarded by the caller; undo the preserved-summary
                # count it provisionally bumped in the pre-check, mirroring the
                # generated counts which are bumped only on compression success.
                # Discard the index too so the set stays consistent with the count,
                # like every sibling compensation site.
                if ve_idx in generation.preserved_summaries:
                    generation.preserved_summaries.discard(ve_idx)
                    generation.summaries_preserved -= 1
            else:
                # Count generated artifacts only AFTER compression succeeds, so a
                # non-atomic compression failure (which discards this entry)
                # cannot inflate the generated counts in the response diagnostic.
                if generation.embeddings.get(ve_idx) is not None:
                    generation.embeddings_generated += 1
                # Only count a summary as GENERATED when a summary model call
                # actually ran for this entry; a reused/preserved summary (no
                # 'summary' task queued, accounted by summaries_preserved)
                # must not be double-counted here, mirroring the single-store
                # generated-vs-preserved split.
                if (
                    'summary' in task_names
                    and generation.summaries.get(ve_idx)
                    and isinstance(generation.summaries[ve_idx], str)
                ):
                    generation.summaries_generated += 1

    return generation


async def generate_store_index_nodes(
    generation: StoreGeneration,
    entries: list[tuple[int, dict[str, Any]]],
) -> None:
    """Build the node summaries of each surviving entry into ``generation.index_nodes``.

    A SEPARATE fenced never-raise pass, isolated from the abort-mandatory
    embedding/summary/compression legs: a node-summary failure never fails an
    entry. None per entry when the feature is disabled, the entry defers its
    generation, OR the entry is a likely deduplication retransmit (gated
    symmetrically with embeddings/summary); None leaves stored rows untouched,
    and a deduplication divergence regenerates the nodes for the diverged text.

    Args:
        generation: The batch's generated outputs.
        entries: ``(position, entry)`` of each entry that survived the generation pass.
    """
    for ve_idx, entry in entries:
        generation.index_nodes[ve_idx] = (
            None
            if generation.likely_duplicates.get(ve_idx) is not None or ve_idx in generation.deferred
            else await generate_index_nodes_with_timeout(entry['text_content'])
        )


async def regenerate_diverged_entries(
    generation: StoreGeneration,
    entries: list[tuple[int, dict[str, Any]]],
    text_content: str,
    *,
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
) -> None:
    """Regenerate the outputs an atomic batch skipped for a text whose deduplication diverged.

    The read-only pre-check skipped generation for a likely duplicate, but the
    transaction inserted a new entry. Every surviving entry sharing the diverged text
    gets the outputs it lacks; a failed embedding or summary propagates its ToolError,
    aborting the atomic batch.

    Args:
        generation: The batch's generated outputs.
        entries: ``(position, entry)`` of each entry that survived the generation pass.
        text_content: The text whose deduplication diverged.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.
    """
    # Only re-run the embedding provider when at least one diverged
    # entry actually lacks embeddings. A node-only reconcile (the
    # nodes_pending gate fired while embeddings are present) must not
    # re-invoke the provider: that wastes a call and a transient
    # failure would spuriously abort the whole atomic batch.
    needs_embeddings = embedding_provider is not None and any(
        generation.embeddings.get(r_ve_idx) is None
        and r_entry['text_content'] == text_content
        for r_ve_idx, r_entry in entries
    )
    if needs_embeddings:
        regenerated = await generate_compression_with_timeout(
            await generate_embeddings_with_timeout(text_content),
        )
        for r_ve_idx, r_entry in entries:
            if (generation.embeddings.get(r_ve_idx) is None
                    and r_entry['text_content'] == text_content):
                generation.embeddings[r_ve_idx] = regenerated
                # Count only embeddings actually produced. With
                # generation disabled the no-op provider returns
                # None, so a node-only reconcile must not inflate
                # the generated count (parity with the single store).
                if regenerated is not None:
                    generation.embeddings_generated += 1
    # A REUSED summary was read from the since-diverged candidate
    # and may describe different text; regenerate it for the
    # diverged text so the divergence INSERT cannot persist a
    # mismatched summary. Clearing the preserved marker also
    # clears the summary_pending reconcile gate on retry. The
    # diverged text is fixed and the prompt varies only by
    # source, so ONE call per distinct source is broadcast to
    # every matching entry -- mirroring the embedding block's
    # compute-once pattern above instead of issuing redundant,
    # sequential LLM round-trips for identical (text, source)
    # pairs on this abort-sensitive path.
    if summary_provider is not None:
        regenerated_summaries: dict[str, str | None] = {}
        for r_ve_idx, r_entry in entries:
            if (r_ve_idx in generation.preserved_summaries
                    and r_entry['text_content'] == text_content):
                r_source = r_entry['source']
                if r_source not in regenerated_summaries:
                    regenerated_summaries[r_source] = await generate_summary_with_timeout(
                        text_content, r_source,
                    )
                generation.summaries[r_ve_idx] = regenerated_summaries[r_source]
                generation.preserved_summaries.discard(r_ve_idx)
                generation.summaries_preserved -= 1
                if generation.summaries[r_ve_idx] is not None:
                    generation.summaries_generated += 1
    # Node summaries for the diverged text were gated off as a
    # likely duplicate; this divergence INSERTs, so regenerate them.
    # Coerce total degradation (None) to [] so the reconcile gate
    # clears on retry: the node layer is never-raise and must not
    # abort the store; a fresh INSERT has no stale node rows.
    regenerated_nodes = await generate_index_nodes_with_timeout(text_content) or []
    for r_ve_idx, r_entry in entries:
        if (generation.index_nodes.get(r_ve_idx) is None
                and r_entry['text_content'] == text_content):
            generation.index_nodes[r_ve_idx] = regenerated_nodes


async def regenerate_diverged_entry(
    generation: StoreGeneration,
    ve_idx: int,
    entry: dict[str, Any],
    text_content: str,
    *,
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
) -> str | None:
    """Regenerate the outputs one non-atomic entry skipped when its deduplication diverged.

    The pre-check skipped generation for a likely duplicate, but this entry's store
    inserted a new entry. A failed regeneration fails ONLY this entry: it is logged, the
    entry's response-counter contributions are reversed, and its error is returned for the
    caller to record, preserving the per-entry isolation of a non-atomic batch.

    Args:
        generation: The batch's generated outputs.
        ve_idx: The entry's position among the validated entries.
        entry: The validated entry.
        text_content: The text whose deduplication diverged.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.

    Returns:
        None when the entry holds its regenerated outputs, else the error to report for it.
    """
    original_idx = entry['index']
    # Only re-run the embedding provider when this entry's embeddings were
    # actually skipped; a node-only reconcile must not re-invoke the provider
    # (wasted call + a transient failure would spuriously fail an otherwise-good
    # entry).
    try:
        if embedding_provider is not None and generation.embeddings.get(ve_idx) is None:
            generation.embeddings[ve_idx] = await generate_compression_with_timeout(
                await generate_embeddings_with_timeout(text_content),
            )
            # Count only embeddings actually produced (the no-op
            # provider returns None when generation is disabled).
            if generation.embeddings[ve_idx] is not None:
                generation.embeddings_generated += 1
    except ToolError as e:
        logger.error(
            'Failed to regenerate embeddings for entry at index %d '
            'after deduplication divergence: %s', original_idx, e,
        )
        generation.discard_counts(ve_idx)
        return format_exception_message(e)
    # A REUSED summary was read from the since-diverged candidate and may
    # describe different text; regenerate it for this entry's text, with the
    # same per-entry isolation. Clearing the preserved marker also clears the
    # summary_pending reconcile gate on retry.
    if summary_provider is not None and ve_idx in generation.preserved_summaries:
        try:
            generation.summaries[ve_idx] = await generate_summary_with_timeout(
                text_content, entry['source'],
            )
        except ToolError as e:
            logger.error(
                'Failed to regenerate summary for entry at index %d '
                'after deduplication divergence: %s', original_idx, e,
            )
            # Reverse every response-counter contribution, including the
            # embeddings regenerated above.
            generation.discard_counts(ve_idx)
            return format_exception_message(e)
        generation.preserved_summaries.discard(ve_idx)
        generation.summaries_preserved -= 1
        if generation.summaries[ve_idx] is not None:
            generation.summaries_generated += 1
    # Node summaries were gated off as a likely duplicate; this entry actually
    # INSERTs, so regenerate its index_tree nodes. Coerce total degradation
    # (None) to [] so the reconcile gate clears on retry: the node layer is
    # never-raise and must not abort this entry's store; a fresh INSERT has no
    # stale rows.
    if generation.index_nodes.get(ve_idx) is None:
        generation.index_nodes[ve_idx] = await generate_index_nodes_with_timeout(
            text_content,
        ) or []
    return None

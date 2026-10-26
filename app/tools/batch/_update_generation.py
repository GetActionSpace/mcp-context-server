"""The generation pass of the update_context_batch tool.

Every output a text change writes is generated here, before any transaction opens: the
embeddings and the summary of each new text, the summaries it leaves stale and marks for
clearing, and the node summaries of the text changes that survive the pass.
``UpdateGeneration`` holds the outputs by the update's position among the validated
updates, with the response counters they feed.
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

from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.tools._generation import generate_compression_with_timeout
from app.tools._generation import generate_embeddings_with_timeout
from app.tools._generation import generate_index_nodes_with_timeout
from app.tools._generation import generate_summary_with_timeout
from app.tools._generation import generation_active

if TYPE_CHECKING:
    from app.embeddings import EmbeddingProvider
    from app.summary import SummaryProvider

logger = logging.getLogger(__name__)


@dataclass
class UpdateGeneration:
    """The generated outputs of a batch update's text changes and the response counters they feed.

    Every map is keyed by the update's position among the validated updates.

    Attributes:
        deferred: The text changes whose generation is left to a job.
        embeddings: Each update's compressed chunk embeddings; None when none were generated.
        summaries: Each update's regenerated summary.
        clear_summaries: The updates whose new text leaves the stored summary stale.
        index_nodes: Each text change's node summaries; an update without text is absent,
            which leaves the stored rows untouched.
        errors: ``(index, context_id, error)`` of each update whose generation failed in a
            non-atomic batch.
        embeddings_generated: Updates holding generated embeddings.
        summaries_generated: Updates holding a regenerated summary.
    """

    deferred: set[int]
    embeddings: dict[int, list[ChunkEmbedding] | None] = field(default_factory=dict[int, list[ChunkEmbedding] | None])
    summaries: dict[int, str | None] = field(default_factory=dict[int, str | None])
    clear_summaries: set[int] = field(default_factory=set[int])
    index_nodes: dict[int, list[IndexNodeRow] | None] = field(default_factory=dict[int, list[IndexNodeRow] | None])
    errors: list[tuple[int, str, str]] = field(default_factory=list[tuple[int, str, str]])
    embeddings_generated: int = 0
    summaries_generated: int = 0

    def discard_counts(self, vu_idx: int) -> None:
        """Reverse a discarded update's response-counter contributions.

        A transaction-phase failure discards an update AFTER the
        generation phase bumped the response counters, so without
        compensation the response message would claim "embeddings
        regenerated" / "summaries regenerated" for updates that modified
        nothing. The decrement criteria mirror the bump criteria exactly:
        one embeddings bump iff the update holds compressed embeddings,
        one summary bump iff it holds a non-empty regenerated summary
        string.

        Args:
            vu_idx: The update's position among the validated updates.
        """
        if self.embeddings.get(vu_idx) is not None:
            self.embeddings_generated -= 1
        summary_value = self.summaries.get(vu_idx)
        if isinstance(summary_value, str) and summary_value:
            self.summaries_generated -= 1


async def generate_update_outputs(
    updates: list[tuple[int, dict[str, Any]]],
    entry_sources: dict[str, str],
    *,
    embedding_provider: 'EmbeddingProvider | None',
    summary_provider: 'SummaryProvider | None',
    min_content_length: int,
    atomic: bool,
) -> UpdateGeneration:
    """Generate the embeddings and summary of each text change, outside any transaction.

    Updates are processed sequentially to limit provider load; each text change runs its
    embedding and summary legs in parallel. An update without text generates nothing, and
    so does a text change asking to defer while a generation provider is active.

    Args:
        updates: ``(position, update)`` of each validated update whose entry exists.
        entry_sources: The source of each updated entry, by context id, which the summary
            prompt varies by.
        embedding_provider: The active embedding provider, or None.
        summary_provider: The active summary provider, or None.
        min_content_length: The shortest text that gets a summary; 0 admits every text.
        atomic: Whether the batch is all-or-nothing.

    Returns:
        The generated outputs, with ``errors`` naming each update whose generation failed.

    Raises:
        ToolError: In atomic mode, on the first update whose generation or compression failed.
    """
    # The text updates whose generation is left to a job: those asking for it while a
    # generation provider is active. Without one there is nothing to defer, and such an
    # update is applied like any other.
    generation_can_defer = generation_active()
    generation = UpdateGeneration(deferred={
        vu_idx for vu_idx, update in updates
        if update['defer_generation'] and update.get('text') is not None and generation_can_defer
    })

    for vu_idx, update in updates:
        original_idx = update['index']
        context_id = update['context_id']
        text_content = update.get('text')

        # No text change -- skip generation entirely
        if text_content is None:
            generation.embeddings[vu_idx] = None
            generation.summaries[vu_idx] = None
            continue

        # A deferring text change generates nothing here; its transaction clears the
        # old text's summary, embeddings and node rows and leaves the new ones to a job.
        if vu_idx in generation.deferred:
            generation.embeddings[vu_idx] = None
            generation.summaries[vu_idx] = None
            generation.clear_summaries.add(vu_idx)
            continue

        tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
        task_names: list[str] = []

        # Embedding task
        if embedding_provider is not None:
            tasks_to_run.append(generate_embeddings_with_timeout(text_content))
            task_names.append('embedding')

        # Summary task (min_content_length pre-check OUTSIDE wrapper)
        if summary_provider is not None:
            if min_content_length > 0 and len(text_content) < min_content_length:
                generation.summaries[vu_idx] = None
                generation.clear_summaries.add(vu_idx)
                logger.info(
                    'Skipping summary for context %s at index %d: '
                    'text length %d < min_content_length %d. Existing summary will be cleared.',
                    context_id, original_idx, len(text_content), min_content_length,
                )
            else:
                tasks_to_run.append(generate_summary_with_timeout(text_content, entry_sources[context_id]))
                task_names.append('summary')
        else:
            # No summary provider at update time (summary generation disabled/absent).
            # The stored summary describes the REPLACED text, so CLEAR it instead of
            # leaving a stale summary -- mirroring the too-short branch above, the
            # single-update path (app/tools/context/update.py), and the stale-embedding / index_tree
            # node-row clears that already run on this same text-change path.
            generation.summaries[vu_idx] = None
            generation.clear_summaries.add(vu_idx)

        if not tasks_to_run:
            generation.embeddings.setdefault(vu_idx, None)
            if vu_idx not in generation.summaries:
                generation.summaries[vu_idx] = None
            continue

        gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        errors: list[tuple[str, BaseException]] = []
        for name, result in zip(task_names, gather_results, strict=True):
            if isinstance(result, BaseException):
                errors.append((name, result))
                logger.error(
                    'Generation failed for %s on context %s at index %d (retries exhausted): %s',
                    name, context_id, original_idx, result,
                )
            elif name == 'embedding':
                generation.embeddings[vu_idx] = cast(list[ChunkEmbedding] | None, result)
            elif name == 'summary':
                generation.summaries[vu_idx] = cast(str | None, result)

        # Summary regeneration that ran but produced nothing (empty -> None)
        # on a text change must CLEAR the now-stale summary (it describes the
        # OLD text), mirroring the too-short branch above and the single-update
        # path. Only when the summary task itself did not error.
        summary_errored = any(name == 'summary' for name, _ in errors)
        if 'summary' in task_names and not summary_errored and generation.summaries.get(vu_idx) is None:
            generation.clear_summaries.add(vu_idx)

        if errors:
            error_details = '; '.join(
                f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
            )
            if atomic:
                raise ToolError(
                    f'Generation failed for context {context_id} at index {original_idx} '
                    f'after exhausting configured retries: {error_details}. No data was modified.',
                )
            generation.errors.append((original_idx, context_id, f'Generation failed: {error_details}'))
            generation.embeddings.setdefault(vu_idx, None)
            generation.summaries.setdefault(vu_idx, None)
        else:
            generation.embeddings.setdefault(vu_idx, None)
            if vu_idx not in generation.summaries:
                generation.summaries[vu_idx] = None

            # Compress regenerated embeddings OUTSIDE the DB transaction
            # (generation-first invariant). No-op when compression is
            # disabled. Atomic mode aborts the batch; non-atomic marks
            # this entry failed and continues.
            try:
                generation.embeddings[vu_idx] = await generate_compression_with_timeout(
                    generation.embeddings.get(vu_idx),
                )
            except ToolError as compress_err:
                if atomic:
                    raise ToolError(
                        f'Compression failed for context {context_id} at '
                        f'index {original_idx}: {compress_err}. '
                        f'No data was modified.',
                    ) from compress_err
                generation.errors.append(
                    (original_idx, context_id,
                     f'Compression failed: {compress_err}'),
                )
                generation.embeddings[vu_idx] = None
                generation.summaries[vu_idx] = None
            else:
                # Count generated artifacts only AFTER compression succeeds, so a
                # non-atomic compression failure (which discards this entry)
                # cannot inflate the generated counts in the response diagnostic.
                if generation.embeddings.get(vu_idx) is not None:
                    generation.embeddings_generated += 1
                if generation.summaries.get(vu_idx) and isinstance(generation.summaries[vu_idx], str):
                    generation.summaries_generated += 1

    return generation


async def generate_update_index_nodes(
    generation: UpdateGeneration,
    updates: list[tuple[int, dict[str, Any]]],
) -> None:
    """Rebuild the node summaries of each surviving text change into ``generation.index_nodes``.

    A SEPARATE fenced never-raise pass, isolated from the abort-mandatory generation:
    a node-summary failure never fails an update. Updates without a text change are
    absent from the map, so ``.get()`` yields None and leaves the node table untouched;
    a deferring text change clears the rows describing the old text.

    Args:
        generation: The batch's generated outputs.
        updates: ``(position, update)`` of each update that survived the generation pass.
    """
    for vu_idx, update in updates:
        new_text = update.get('text')
        if vu_idx in generation.deferred:
            generation.index_nodes[vu_idx] = []
        elif new_text is not None:
            rebuilt = await generate_index_nodes_with_timeout(new_text)
            # Text changed: a None result must CLEAR the rows describing the old text
            # ([]) because navigate_context attaches stored summaries by heading-slug
            # node_id, so a retained slug would otherwise receive its pre-edit summary.
            # The clear is UNCONDITIONAL (not gated on node_summaries_enabled) so a
            # disable/edit/re-enable cycle cannot resurface stale rows; it covers the
            # provider-removed case (None for lack of a provider) and total degradation,
            # mirroring the single-update path and the provider-independent
            # embedding/summary clears. replace_nodes_for_context pre-checks table
            # existence, so clearing when the table is absent is a safe no-op.
            generation.index_nodes[vu_idx] = [] if rebuilt is None else rebuilt

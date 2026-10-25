"""Embedding, compression, and summary generation that runs outside the transaction.

Owns the concurrency limiters for the embedding provider, the summary model,
compression encoding, and index_tree node fan-out; the timeout-bounded
generators for embeddings, compression, the flat summary, and per-node
summaries; and ``run_generation``, which overlaps those legs on the
single-entry store and update paths.
"""

import asyncio
import logging
import time
from typing import Any

from fastmcp.exceptions import ToolError

from app.embeddings.batching import BatchSizer
from app.embeddings.batching import EmbeddingBatchError
from app.embeddings.batching import EmbeddingBatchTimeoutError
from app.embeddings.batching import create_batch_sizer
from app.embeddings.batching import embed_in_batches
from app.embeddings.retry import compute_embedding_total_timeout
from app.errors import format_exception_message
from app.metadata_types import sanitize_pg_unstorable_text
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.services.outline_service import OutlineNode
from app.services.outline_service import parse_outline
from app.services.text_lines import should_offload_line_scan
from app.settings import get_settings
from app.startup import get_chunking_service
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.summary.instructions import resolve_index_tree_node_summary_prompt
from app.summary.retry import compute_summary_total_timeout

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Concurrency limiters for embedding / summary / compression generation
# ---------------------------------------------------------------------------
#
# All three semaphores are constructed at module import time. asyncio.Semaphore
# has been parameterless (no ``loop`` argument) since Python 3.10, so its
# construction does NOT require a running event loop. Constructing at module
# scope is simpler than the prior lazy-init helpers and gives every caller a
# stable reference for the lifetime of the process.
#
# The semaphores are intentionally separate, one per physical resource:
#   * ``_embedding_semaphore``: bounds outbound HTTP concurrency to the
#     embedding provider. A chunked document holds it for one batch at a time
#     (``app.embeddings.batching.embed_in_batches``), so a waiting caller is
#     served after the current batch of a running document.
#   * ``_summary_model_semaphore``: bounds outbound concurrency to the single
#     physical SUMMARY model. It is acquired by BOTH the flat document summary
#     (``generate_summary_with_timeout``) AND every per-node index_tree summary
#     (``_summarize_node``) -- both hit the same model, so ONE shared budget
#     caps global summary-model concurrency at ``SUMMARY_MAX_CONCURRENT`` no
#     matter how the flat and node passes overlap. That is what protects a small
#     local model (e.g. Ollama) from the overload the 3->2 de-tune fixed.
#   * ``_compression_semaphore``: bounds CPU-bound encoding offloaded via
#     ``asyncio.to_thread``; contention is for the GIL / CPU, not the event
#     loop, so a separate budget applies.
#   * ``_node_summary_semaphore``: a node-task LAUNCH / fan-out cap (NOT a second
#     model budget). It bounds how many ``_summarize_node`` coroutines are
#     in-flight at once so a many-heading document cannot create an unbounded
#     fan-out; the inner ``_summary_model_semaphore`` is what actually gates the
#     model call. Sized by ``INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT``.

_embedding_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.embedding.max_concurrent,
)


_summary_model_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.summary.max_concurrent,
)


_compression_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.compression.max_concurrent,
)


_node_summary_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.index_tree.max_concurrent,
)


def _reset_embedding_semaphore() -> None:
    """Rebind the embedding semaphore against the current settings value.

    Test fixtures that mutate ``settings.embedding.max_concurrent`` between
    cases call this to ensure the next ``async with _embedding_semaphore``
    block uses the freshly configured limit.
    """
    global _embedding_semaphore
    _embedding_semaphore = asyncio.Semaphore(settings.embedding.max_concurrent)


def _reset_summary_model_semaphore() -> None:
    """Rebind the shared summary-model semaphore against current settings.

    Test fixtures that mutate ``settings.summary.max_concurrent`` between cases
    call this so the next ``async with _summary_model_semaphore`` block -- used
    by BOTH the flat document summary and every per-node index_tree summary --
    uses the freshly configured limit.
    """
    global _summary_model_semaphore
    _summary_model_semaphore = asyncio.Semaphore(settings.summary.max_concurrent)


def _reset_compression_semaphore() -> None:
    """Rebind the compression semaphore against the current settings value.

    Test fixtures that mutate ``settings.compression.max_concurrent``
    between cases call this to ensure the next
    ``async with _compression_semaphore`` block uses the freshly
    configured limit.
    """
    global _compression_semaphore
    _compression_semaphore = asyncio.Semaphore(settings.compression.max_concurrent)


def _reset_node_summary_semaphore() -> None:
    """Rebind the node-summary semaphore against the current settings value.

    Test fixtures that mutate ``settings.index_tree.max_concurrent`` between
    cases call this to ensure the next ``async with _node_summary_semaphore``
    block uses the freshly configured limit.
    """
    global _node_summary_semaphore
    _node_summary_semaphore = asyncio.Semaphore(settings.index_tree.max_concurrent)


# The embedding batch sizer, built from the settings on first use (see
# ``_get_batch_sizer``); ``app.embeddings.batching`` documents the sizing.
_batch_sizer: BatchSizer | None = None


def _get_batch_sizer() -> BatchSizer:
    """Return the process-wide embedding batch sizer, building it on first use.

    One sizer serves every document, so the auto sizer's seconds-per-chunk
    estimate carries from one document to the next.

    Returns:
        The sizer ``EMBEDDING_BATCH_SIZE`` selects.
    """
    global _batch_sizer
    if _batch_sizer is None:
        _batch_sizer = create_batch_sizer(settings.embedding)
    return _batch_sizer


def _reset_batch_sizer() -> None:
    """Drop the batch sizer so the next document builds one from the current settings.

    Test fixtures that mutate ``settings.embedding.batch_size`` or
    ``settings.embedding.batch_target_s`` between cases call this.
    """
    global _batch_sizer
    _batch_sizer = None


# Explicit re-export so type checkers do NOT flag the reset helpers as unused.
# These are called only by test fixtures that need to rebind the module-level
# semaphores and the batch sizer against patched ``settings`` values between
# cases; production code uses them directly without rebinding.
_RESET_HELPERS_EXPORT = (
    _reset_embedding_semaphore,
    _reset_summary_model_semaphore,
    _reset_compression_semaphore,
    _reset_node_summary_semaphore,
    _reset_batch_sizer,
)


# ---------------------------------------------------------------------------
# Embedding and summary generation with timeout
# ---------------------------------------------------------------------------


async def generate_embeddings_with_timeout(text: str) -> list[ChunkEmbedding] | None:
    """Generate the embeddings of a text, bounded per provider request.

    Runs BEFORE any database transaction (generation-first): when generation is
    enabled and fails, the caller saves nothing. A chunked text is embedded in
    batches by ``embed_in_batches``: each batch holds one slot of the embedding
    semaphore and runs under ``compute_embedding_total_timeout()``, so the
    provider's retries and the total timeout apply per batch and a caller waiting
    for a slot is served after the current batch of a running document. With
    chunking disabled the whole text is one ``embed_query`` request under one slot
    and the same timeout.

    Used by store_context, update_context, store_context_batch,
    update_context_batch, and the migration CLI's embedding modes.

    Args:
        text: Text content to generate embeddings for.

    Returns:
        One ChunkEmbedding per chunk with its vector and boundaries, or None when
        no embedding provider is configured.

    Raises:
        ToolError: If embedding generation fails or a request exceeds the total
            timeout; a chunked text's message names the batch and the progress made.
    """
    embedding_provider = get_embedding_provider()
    if embedding_provider is None:
        return None

    total_timeout = compute_embedding_total_timeout()
    try:
        chunking_service = get_chunking_service()
        logger.debug(
            f'Chunking service state: service={chunking_service}, '
            f'enabled={chunking_service.is_enabled if chunking_service else "N/A"}',
        )

        if chunking_service is not None and chunking_service.is_enabled:
            # Chunked embedding for long documents. split_text (->
            # RecursiveCharacterTextSplitter.create_documents) is pure CPU over
            # unbounded entry text and is EXPENSIVE per character -- roughly a
            # microsecond each, so a plain 1MB document costs the better part of a
            # second. Gating on a large fixed character threshold left exactly that
            # case running inline on the event loop. The real cost switch is the
            # splitter's own fast path: at or below chunk_size split_text returns
            # immediately (one chunk, no recursion), and above it the recursive
            # split runs in full. Offload precisely when that recursion happens --
            # the thread hop is negligible next to the provider round trip this leg
            # is about to await anyway.
            if len(text) > chunking_service.chunk_size:
                chunks = await asyncio.to_thread(chunking_service.split_text, text)
            else:
                chunks = chunking_service.split_text(text)
            sizer = _get_batch_sizer()
            logger.info(
                'Generating embeddings: text_len=%d, chunks=%d, batch_size=%s', len(text), len(chunks), sizer.mode,
            )
            embeddings = await embed_in_batches(
                embedding_provider,
                [chunk.text for chunk in chunks],
                sizer=sizer,
                semaphore=_embedding_semaphore,
                batch_timeout_s=total_timeout,
            )
            return [
                ChunkEmbedding(
                    embedding=emb,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                )
                for emb, chunk in zip(embeddings, chunks, strict=True)
            ]
        # Single embedding (chunking disabled)
        logger.info(f'Generating single embedding: text_len={len(text)}')
        async with _embedding_semaphore:
            embedding = await asyncio.wait_for(embedding_provider.embed_query(text), timeout=total_timeout)
        logger.info('Single embedding generated')
        return [ChunkEmbedding(embedding=embedding, start_index=0, end_index=len(text))]

    # CRITICAL: every failure leaves as ToolError so the caller saves nothing.
    except EmbeddingBatchTimeoutError as e:
        position = e.position
        raise ToolError(
            f'Embedding generation exceeded total timeout ({total_timeout:.0f}s) on {position.batch_label} '
            f'({position.chunk_label}; {position.progress_label}). '
            f'This may indicate the embedding provider is overloaded or unreachable.',
        ) from None
    except EmbeddingBatchError as e:
        position = e.position
        raise ToolError(
            f'Embedding generation failed: {format_exception_message(e.cause)} '
            f'({position.batch_label}, {position.chunk_label}; {position.progress_label})',
        ) from e
    except TimeoutError:
        raise ToolError(
            f'Embedding generation exceeded total timeout ({total_timeout:.0f}s). '
            f'This may indicate the embedding provider is overloaded or unreachable.',
        ) from None
    except Exception as e:
        raise ToolError(f'Embedding generation failed: {format_exception_message(e)}') from e


async def generate_compression_with_timeout(
    chunk_embeddings: list[ChunkEmbedding] | None,
) -> list[ChunkEmbedding] | None:
    """Compress each chunk's embedding into a bytes payload.

    Runs OUTSIDE any DB transaction, preserving the generation-first
    transactional-integrity invariant: when compression fails the storage
    write does not happen and the entry is not persisted.

    When ENABLE_EMBEDDING_COMPRESSION is false this is a no-op that returns
    the input unchanged so callers can wire the helper unconditionally.
    When true it calls the active provider's ``encode_sync`` for each chunk
    inside a worker thread (``asyncio.to_thread``) bounded by the
    compression semaphore, returning a fresh ``ChunkEmbedding`` list with
    the ``payload`` field populated.

    Args:
        chunk_embeddings: Embeddings returned by
            :func:`generate_embeddings_with_timeout`. ``None`` is passed
            through unchanged (no embeddings to compress).

    Returns:
        The same list of ``ChunkEmbedding`` objects when compression is
        disabled or ``chunk_embeddings is None``; a fresh list with
        ``payload`` populated otherwise.

    Raises:
        ToolError: If compression provider construction or any encode call
            fails. The transactional write is aborted by the propagating
            exception.
    """
    if not settings.compression.enabled:
        return chunk_embeddings

    if chunk_embeddings is None:
        return None

    # Defer provider import until enabled to keep numpy out of the import
    # graph for installations that skipped the compression extra. The
    # cached helper unifies provider construction across read (search)
    # and write (encode) paths: both reuse the same rotation matrix and
    # codebook arrays per process.
    from app.compression import get_cached_compression_provider

    try:
        provider = await get_cached_compression_provider()
    except Exception as e:
        raise ToolError(
            f'Compression provider initialization failed: '
            f'{format_exception_message(e)}',
        ) from e

    async def _encode_one(chunk: ChunkEmbedding) -> ChunkEmbedding:
        # Acquire one semaphore permit per encode call so the configured
        # COMPRESSION_MAX_CONCURRENT limit governs in-flight CPU work
        # accurately. Wrapping the outer asyncio.gather() would let an
        # N-chunk batch run all N encodes under one permit, bypassing
        # the bound. Mirrors the established embedding/summary semaphore
        # pattern.
        async with _compression_semaphore:
            try:
                # Local import keeps numpy out of the hot import graph; this branch
                # only executes when compression is enabled (extra installed). Both the
                # import and the array construction sit INSIDE the guard so every
                # failure this leg can produce leaves as ToolError: the store paths
                # guard the compression call with `except ToolError`, and a raw
                # exception escaping from here would slip past that guard and abort a
                # whole non-atomic batch instead of failing the one entry.
                import numpy as np

                vector = np.asarray([chunk.embedding], dtype=np.float32)
                payload_bytes = await asyncio.to_thread(provider.encode_sync, vector)
            except Exception as e:
                raise ToolError(
                    f'Compression encode failed: {format_exception_message(e)}',
                ) from e
        return ChunkEmbedding(
            embedding=chunk.embedding,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            payload=payload_bytes,
        )

    # return_exceptions=True keeps the fan-out structured: a bare gather
    # would propagate the first encode failure while the sibling tasks kept
    # running detached past the request (run_generation's finally cancels
    # only the three top-level legs and cannot reach these children).
    # Awaiting every child first, then raising, matches the abort loop in
    # run_generation and every other fan-out on the store paths.
    results = await asyncio.gather(
        *[_encode_one(c) for c in chunk_embeddings],
        return_exceptions=True,
    )
    encoded: list[ChunkEmbedding] = []
    for result in results:
        if isinstance(result, BaseException):
            raise result
        encoded.append(result)
    return encoded


async def generate_summary_with_timeout(text: str, source: str) -> str | None:
    """Generate summary with concurrency limiting and total timeout.

    Wraps summary_provider.summarize() with:
    - Concurrency-limited access via summary semaphore
    - Total timeout computed from retry settings
    - ToolError on timeout for clear client feedback

    Used by all four tools: store_context, update_context,
    store_context_batch, and update_context_batch.

    Args:
        text: Text content to generate summary for.
        source: Source type ('user' or 'agent').

    Returns:
        Summary string, or None if summary provider is not configured.

    Raises:
        ToolError: If summary generation times out or fails. EVERY non-cancellation
            failure is normalized to ToolError, matching the embedding helper's
            contract (see the note in the body).
    """
    summary_provider = get_summary_provider()
    if summary_provider is None:
        return None

    total_timeout = compute_summary_total_timeout()
    try:
        logger.info('Generating summary: text_len=%d', len(text))
        async with _summary_model_semaphore:
            result = await asyncio.wait_for(
                summary_provider.summarize(text, source),
                timeout=total_timeout,
            )
        # Normalize empty/whitespace-only summaries to None
        if not result.strip():
            logger.warning('Summary provider returned empty/whitespace-only response, treating as None')
            return None
        # The summary is model-generated, not client-supplied: a stray NUL or unpaired
        # surrogate in the provider's output would store on SQLite yet abort the
        # PostgreSQL bind inside the (abort-mandatory) transaction, charging the breaker.
        # Repair rather than reject -- the client's own text is valid and must not be
        # refused for a provider quirk.
        sanitized = sanitize_pg_unstorable_text(result)
        logger.info('Summary generated: text_len=%d, summary_len=%d', len(text), len(sanitized))
        return sanitized
    except TimeoutError:
        raise ToolError(
            f'Summary generation exceeded total timeout ({total_timeout:.0f}s). '
            f'This may indicate the summary provider is overloaded or unreachable.',
        ) from None
    except Exception as e:
        # Normalize EVERY provider failure to ToolError, exactly as the embedding
        # leg does (generate_embeddings_with_timeout wraps its own body). Without this
        # the retry layer's SummaryTimeoutError / SummaryRetryExhaustedError and the
        # providers' bare RuntimeError / ValueError escaped raw, so callers that
        # isolate a per-entry failure with ``except ToolError`` -- the non-atomic
        # batch reconcile path -- let a routine provider outage escape their loop and
        # abort the whole request, discarding sibling results already committed. One
        # error contract for both abort-mandatory legs keeps that isolation honest and
        # makes the single-entry and batch error messages consistent.
        # ``except Exception`` deliberately does NOT catch ``CancelledError`` (a
        # ``BaseException``), so run_generation's leg cancellation still propagates.
        raise ToolError(f'Summary generation failed: {format_exception_message(e)}') from e


async def generate_index_nodes_with_timeout(text: str) -> list[IndexNodeRow] | None:
    """Build index_tree node rows with per-node LLM summaries (NEVER raises).

    The code-derived outline is always parsed (pure CPU). Each heading section
    long enough to warrant one is summarized via the existing summary provider's
    ``summarize_with_prompt`` with a dedicated short prompt, bounded by a per-node
    timeout and the node-summary semaphore. This is the additive, fenced layer: a
    provider failure or timeout omits that node's summary and NEVER aborts the
    store -- the deliberate contrast with the abort-mandatory
    embedding/summary/compression helpers above.

    TOTAL work is bounded, not just concurrency: at most
    INDEX_TREE_NODE_SUMMARY_MAX_NODES sections are summarized (the shallowest and
    longest first, so the outline degrades gracefully), they are processed in
    bounded chunks rather than one unbounded gather, and the whole pass runs under
    the INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S aggregate budget.

    Args:
        text: The entry's full text content.

    Returns:
        ``None`` -- meaning "leave the node table untouched" -- when per-node
        summaries are disabled, no summary provider is configured, OR every
        attempted per-node summary failed/timed out (TOTAL degradation: a
        transient provider outage must NOT wipe previously-good stored rows on
        replace). Otherwise the list of node rows that received a summary --
        possibly empty when no section qualified (no headings, or all sections
        below the minimum length), which legitimately clears stale rows on replace.
    """
    if not settings.index_tree.node_summaries_enabled:
        return None

    provider = get_summary_provider()
    if provider is None:
        # Per-node summaries reuse the summary provider; with none configured the
        # feature is inert, so leave the node table untouched (None = no write).
        return None

    try:
        # parse_outline is pure CPU whose cost tracks LINE count over unbounded entry
        # text; offload a large OR line-dense entry to a worker thread so it cannot pin
        # the event loop (see should_offload_line_scan), matching the read-path discipline.
        if should_offload_line_scan(text):
            root = await asyncio.to_thread(parse_outline, text)
        else:
            root = parse_outline(text)
    except Exception as e:  # defensive: parsing is pure CPU and should not fail
        logger.warning('Index-tree outline parse failed; skipping node summaries: %s', e)
        return []

    nodes: list[OutlineNode] = []
    stack = list(root.children)
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)

    if not nodes:
        return []

    min_len = settings.index_tree.min_content_length
    timeout = settings.index_tree.timeout_s
    prompt = resolve_index_tree_node_summary_prompt()

    # A summary is ATTEMPTED only for sections that clear the minimum length;
    # shorter sections are deliberately skipped (not a failure). Filtering here
    # rather than inside the worker means an entry with hundreds of thousands of
    # tiny headings never materializes a task per heading just to return None.
    # len(section) == char_end - char_start (offsets are code points).
    eligible = [node for node in nodes if (node.char_end - node.char_start) >= min_len]
    if not eligible:
        # Nothing qualified: return [] so a replace legitimately clears stale rows.
        return []

    # Bound TOTAL work, not just concurrency. The semaphores cap how many calls run
    # at once and asyncio.wait_for caps each one, but neither bounds HOW MANY happen:
    # a heading-dense entry would otherwise hold the single summary model for one
    # store_context request for as long as its section count demands. When more
    # sections qualify than the cap, prefer the shallowest (most structurally
    # significant) and, within a level, the longest sections, then restore the
    # original traversal order so the stored rows stay deterministic.
    max_nodes = settings.index_tree.max_nodes
    if len(eligible) > max_nodes:
        logger.info(
            'Index-tree node summaries: %d eligible section(s) exceeds the cap of %d; '
            'summarizing the shallowest and longest sections only.',
            len(eligible), max_nodes,
        )
        # Rank by POSITION, never by node identity: OutlineNode is a frozen
        # dataclass holding its children, so hashing one hashes the whole subtree
        # and a set membership test over a large outline would be quadratic.
        ranked = sorted(
            range(len(eligible)),
            key=lambda i: (eligible[i].level, -(eligible[i].char_end - eligible[i].char_start)),
        )
        eligible = [eligible[i] for i in sorted(ranked[:max_nodes])]

    attempted = len(eligible)

    async def _summarize_node(node: OutlineNode) -> IndexNodeRow | None:
        # Eligibility (section length >= min_content_length) is pre-filtered above.
        section = text[node.char_start:node.char_end]
        try:
            # Outer acquire = node-task fan-out cap (bounds how many node
            # coroutines run at once). Inner acquire = the SHARED summary-model
            # budget, so per-node calls and the flat document summary together
            # never exceed SUMMARY_MAX_CONCURRENT on the one physical model.
            async with _node_summary_semaphore, _summary_model_semaphore:
                result = await asyncio.wait_for(
                    provider.summarize_with_prompt(section, prompt),
                    timeout=timeout,
                )
        except Exception as e:
            logger.warning('Index-tree node summary failed for %s (skipped): %s', node.node_id, e)
            return None
        # Repair a model-emitted NUL/unpaired surrogate before it binds into
        # context_index_nodes.node_summary (same abort-mandatory PostgreSQL bind as
        # the flat summary); an all-NUL result sanitizes to empty and is skipped.
        summary = sanitize_pg_unstorable_text(result.strip())
        if not summary:
            return None
        return IndexNodeRow(
            node_id=node.node_id,
            level=node.level,
            ordinal=node.ordinal,
            title=node.title,
            node_summary=summary,
            char_start=node.char_start,
            char_end=node.char_end,
        )

    # Process in bounded chunks under an aggregate wall-clock deadline instead of one
    # mega-gather: the chunking keeps the number of live tasks proportional to the
    # fan-out cap rather than to the section count, and the deadline stops a
    # pathological entry from stretching a single store indefinitely (each per-node
    # wait_for bounds ONE call, and their sum is unbounded without this). Whatever was
    # produced before the deadline is kept -- this leg never aborts a store.
    #
    # The deadline bounds the WORK, not just the gaps between chunks: each chunk's
    # gather runs under the REMAINING budget. Testing the clock only between chunks
    # bounds nothing, because nothing bounds the chunk itself -- chunk_size is
    # max(fan-out * 4, 16) while the effective model concurrency is the smaller
    # shared SUMMARY_MAX_CONCURRENT budget, so one chunk serializes into several
    # waves of per-node timeouts and overruns the aggregate budget by that multiple;
    # and an entry with no more sections than one chunk holds yields a single
    # iteration whose only deadline test happens before any work, where it can never
    # be true, leaving the budget inert for the common case.
    # _summarize_node never raises; return_exceptions is a defensive backstop so a
    # surprise (e.g. cancellation of a child) cannot turn into a store-aborting raise.
    chunk_size = max(settings.index_tree.max_concurrent * 4, 16)
    deadline = time.monotonic() + settings.index_tree.total_timeout_s
    rows: list[IndexNodeRow] = []
    for start in range(0, attempted, chunk_size):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                'Index-tree node summaries: aggregate budget of %.0fs expired after %d of '
                '%d section(s); keeping the summaries produced so far.',
                settings.index_tree.total_timeout_s, start, attempted,
            )
            break
        chunk = eligible[start:start + chunk_size]
        node_tasks = [asyncio.create_task(_summarize_node(node)) for node in chunk]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*node_tasks, return_exceptions=True),
                timeout=remaining,
            )
        except TimeoutError:
            # The budget expired mid-chunk. Cancel AND await the still-running node
            # calls so none outlives this pass holding a shared summary-model permit,
            # then keep every row that did finish -- the leg is never-raise, so an
            # expired budget degrades the outline instead of failing the store.
            for node_task in node_tasks:
                node_task.cancel()
            settled = await asyncio.gather(*node_tasks, return_exceptions=True)
            rows.extend(result for result in settled if isinstance(result, IndexNodeRow))
            logger.warning(
                'Index-tree node summaries: aggregate budget of %.0fs expired during the '
                'section(s) at offset %d of %d; cancelling the outstanding node summaries '
                'and keeping the ones produced so far.',
                settings.index_tree.total_timeout_s, start, attempted,
            )
            break
        rows.extend(result for result in results if isinstance(result, IndexNodeRow))

    # TOTAL degradation: sections were eligible and summaries attempted, but every
    # one failed/timed out. Return None so callers PRESERVE existing stored rows
    # (replace_nodes_for_context with None is a no-op) instead of wiping them.
    if attempted > 0 and not rows:
        logger.warning(
            'Index-tree node summaries: all %d attempted section(s) failed; '
            'preserving any existing stored rows (skipping replace).',
            attempted,
        )
        return None

    return rows


def node_layer_active() -> bool:
    """Whether the index_tree per-node summary layer is ACTIVE (would attempt work).

    Mirrors the activation gate at the top of
    :func:`generate_index_nodes_with_timeout`: the feature toggle is on AND a
    summary provider is configured. Used on the STORE / DEDUPLICATION pre-check
    paths to set the ``nodes_pending`` reconcile flag: the read-only pre-check
    skips generation, so ``nodes_pending`` must record whether node work WOULD
    have been attempted -- which requires both the feature toggle on AND a
    provider present (no provider means no node call was made, hence nothing to
    reconcile).

    NOTE: the TEXT-CHANGE update stale-node clear does NOT use this helper. It
    is UNCONDITIONAL (a None node result on a text change becomes an empty
    list regardless of any toggle or provider), so stale rows describing the
    old text can never survive the edit -- not even through a
    disable/edit/re-enable cycle -- and ``navigate_context`` can never
    mis-attach an old section summary to a new section sharing a reused
    heading slug. ``replace_nodes_for_context`` pre-checks table existence,
    making the clear a safe no-op when the node table is absent.

    Returns:
        True when per-node summaries are enabled and a summary provider exists.
    """
    return settings.index_tree.node_summaries_enabled and get_summary_provider() is not None


async def embed_then_compress(text: str) -> list[ChunkEmbedding] | None:
    """Generate embeddings then compress them, as ONE abort-mandatory leg.

    Compression has a hard data dependency on the embeddings, so chaining keeps
    that dependency while letting the whole (embedding -> compression) leg
    overlap the concurrently-running summary/node leg in store/update instead of
    serializing after it. Both steps are generation-first: a failure propagates
    so nothing is saved. Compression is a no-op passthrough when
    ENABLE_EMBEDDING_COMPRESSION is false.

    Returns:
        The compressed ``ChunkEmbedding`` list, or ``None`` when no embedding
        provider is configured.
    """
    chunk_embeddings = await generate_embeddings_with_timeout(text)
    return await generate_compression_with_timeout(chunk_embeddings)


async def _nodes_after_summary(
    summary_task: asyncio.Task[str | None] | None,
    text: str,
) -> list[IndexNodeRow] | None:
    """Generate index_tree node summaries AFTER the flat summary completes.

    Awaiting the flat-summary task first gives the ABORT-MANDATORY flat summary
    strict precedence on the shared summary-model budget, so the never-raise node
    summaries can never starve it (no latency inversion). If the flat summary
    failed there is no store to enrich, so node generation is skipped. Never
    raises on a provider error (mirrors ``generate_index_nodes_with_timeout``); a
    cancellation still propagates.

    Returns:
        The node rows, or ``None`` when nodes are skipped/disabled or the flat
        summary failed.

    Raises:
        asyncio.CancelledError: Propagated (not swallowed) if this leg is cancelled.
    """
    if summary_task is not None:
        try:
            await summary_task
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
    return await generate_index_nodes_with_timeout(text)


async def run_generation(
    text: str,
    source: str,
    *,
    run_embedding: bool,
    run_summary: bool,
    run_nodes: bool,
) -> tuple[list[ChunkEmbedding] | None, str | None, list[IndexNodeRow] | None]:
    """Run the embedding->compression, flat-summary, and node-summary legs concurrently.

    The embedding->compression leg (embedding model + CPU) and the summary legs
    (summary model) use disjoint resources, so they overlap genuinely -- this is
    what removes the node-summary serial tail and the post-gather compression wait
    from store/update latency. The node-summary leg starts only AFTER the flat
    summary finishes, keeping the abort-mandatory flat summary's precedence on the
    shared summary-model budget (no latency inversion).

    The embedding leg and the flat summary are ABORT-MANDATORY: both are awaited
    and ALL their errors are collected, so a failure reports every abort-mandatory
    leg deterministically. On EVERY exit path -- a normal return, the combined
    abort ToolError, OR an outer cancellation (MCP client disconnect / request
    timeout) landing on the abort-legs gather -- the ``finally`` cancels and awaits
    every created task that is not yet done. So no in-flight summary-model or
    embedding call outlives the request: in particular the never-raise node leg,
    which when ``run_summary=False`` does NOT transitively cancel via the flat
    summary, can never be orphaned holding the shared summary-model permit.

    Returns:
        ``(chunk_embeddings, summary_text, index_nodes)``; any leg that was not
        requested yields ``None``.

    Raises:
        ToolError: If an abort-mandatory leg (embeddings, compression, or the
            flat summary) fails after exhausting its configured retries; the
            message names every failed leg.
    """
    embed_task: asyncio.Task[list[ChunkEmbedding] | None] | None = None
    summary_task: asyncio.Task[str | None] | None = None
    node_task: asyncio.Task[list[IndexNodeRow] | None] | None = None

    try:
        if run_embedding:
            embed_task = asyncio.create_task(embed_then_compress(text))
        if run_summary:
            summary_task = asyncio.create_task(generate_summary_with_timeout(text, source))
        if run_nodes:
            node_task = asyncio.create_task(_nodes_after_summary(summary_task, text))

        # Await the ABORT-MANDATORY legs, collecting every error (return_exceptions so
        # one failure does not hide another); the never-raise node leg is NOT awaited
        # here so its timeouts can never delay an abort.
        abort_legs: list[tuple[str, asyncio.Task[Any]]] = []
        if embed_task is not None:
            abort_legs.append(('embedding', embed_task))
        if summary_task is not None:
            abort_legs.append(('summary', summary_task))

        errors: list[str] = []
        if abort_legs:
            # Inspect the GATHER RESULTS, not Task.exception(): a task whose
            # coroutine ended with CancelledError is marked cancelled, and
            # Task.exception() then RAISES CancelledError instead of returning
            # it -- so an exception()-based loop can never skip a cancelled leg
            # (the tolerance the CancelledError check intends). gather with
            # return_exceptions=True hands back the CancelledError as a value,
            # which is safe to type-check.
            leg_outcomes = await asyncio.gather(
                *(task for _, task in abort_legs), return_exceptions=True,
            )
            for (name, _task), outcome in zip(abort_legs, leg_outcomes, strict=True):
                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                    errors.append(f'{name}: {type(outcome).__name__}: {outcome}')

        if errors:
            # Abort-mandatory failure: surface a combined, deterministic error
            # naming every failed leg. The never-raise node leg (and any other
            # in-flight leg) is cancelled and awaited by the ``finally`` below, so
            # no in-flight summary-model call outlives the failed request.
            raise ToolError(
                'Generation failed after exhausting configured retries: ' + '; '.join(errors),
            )

        chunk_embeddings = embed_task.result() if embed_task is not None else None
        summary_text = summary_task.result() if summary_task is not None else None
        # The index_tree node leg is contractually NEVER-RAISE: a node-summary
        # failure or timeout must never abort a store (None preserves existing
        # node rows). _nodes_after_summary already swallows its own non-Cancelled
        # exceptions, so this guard is defense-in-depth that keeps the structural
        # generation-first guarantee intact against any future regression in the
        # node helpers -- a surprise node-leg exception is coerced to None rather
        # than aborting an otherwise-successful store.
        # ``except Exception`` deliberately does NOT catch ``CancelledError``
        # (it subclasses ``BaseException``), so an inner/outer cancellation still
        # propagates and is cleaned up by the ``finally`` below.
        index_nodes: list[IndexNodeRow] | None = None
        if node_task is not None:
            try:
                index_nodes = await node_task
            except Exception:
                logger.warning(
                    'Index-tree node leg raised unexpectedly; preserving existing '
                    'node rows (None).',
                    exc_info=True,
                )
                index_nodes = None
        return chunk_embeddings, summary_text, index_nodes
    finally:
        # Guarantee NO created task outlives this coroutine on ANY exit path. On a
        # normal return every task is already done (no-op). On the abort ToolError
        # the node leg may still be running. On OUTER cancellation, CancelledError
        # propagates straight out of the abort-legs gather BEFORE the final
        # ``await node_task`` runs, so without this cleanup the node leg -- when
        # run_summary=False it never transitively cancels via the flat summary --
        # would be orphaned and keep holding its shared summary-model permit,
        # progressively starving all summary generation. Cancelling and awaiting
        # every not-done task here always releases the embedding/summary-model
        # permits and ensures no orphaned model call survives the request.
        pending: list[asyncio.Task[Any]] = [
            task for task in (embed_task, summary_task, node_task) if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

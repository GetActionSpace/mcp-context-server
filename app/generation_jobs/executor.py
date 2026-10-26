"""Run one claimed generation job: generate the outputs its entry lacks and store them.

The executor reads the entry's text and which outputs it already holds, and runs the
embedding, summary and node-summary legs for the outputs that are missing and whose
provider is active. While they run, a lease keeper renews the claim's lease; a renewal that
finds the claim gone cancels the generation at once, because the job was replaced, reset
or deleted meanwhile.

The outputs are stored in one transaction that first writes the entry row, only while the
entry still holds the text that was read, then deletes the job under its claim, then writes
the embeddings and the node rows. A text changed, an entry deleted or a claim lost during
the generation rolls that transaction back and discards the outputs. Any other failure
spends the attempt: the job returns to the queue after a backoff, or fails after its last
attempt, which is logged at ERROR with the remedy.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from app.backends.base import BackendAccess
from app.errors import ControlFlowError
from app.errors import format_exception_message
from app.repositories.context_repository.records import GenerationSource
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.generation_job_repository import ClaimedJob
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from app.startup import get_embedding_provider
from app.tools._generation import node_layer_active
from app.tools._generation import run_generation
from app.tools._generation import summary_applies
from app.tools._transactions import write_generation_outputs

if TYPE_CHECKING:
    from app.repositories import RepositoryContainer

logger = logging.getLogger(__name__)
settings = get_settings()

# The seconds a claim holds its job before another worker may claim it again.
LEASE_S = 120

# The seconds between two renewals of a running job's lease.
LEASE_RENEWAL_INTERVAL_S = 30.0

type _Outputs = tuple[list[ChunkEmbedding] | None, str | None, list[IndexNodeRow] | None]


class JobOutcome(StrEnum):
    """What running a claimed job did."""

    COMPLETED = 'completed'  # the outputs were stored and the job deleted
    DISCARDED = 'discarded'  # the job, its entry or its text changed meanwhile; nothing was stored
    RETRYING = 'retrying'  # the attempt failed; the job is claimable again after a backoff
    FAILED = 'failed'  # the last attempt failed; the job stays failed
    UNRECORDED = 'unrecorded'  # the attempt failed and so did recording it; the job's lease decides


@dataclass(frozen=True, slots=True)
class _Legs:
    """The generation legs a job runs."""

    embedding: bool
    summary: bool
    nodes: bool

    def any(self) -> bool:
        """Whether any leg runs."""
        return self.embedding or self.summary or self.nodes

    def describe(self) -> str:
        """Name the legs that run, or say that none does."""
        legs = (('embeddings', self.embedding), ('summary', self.summary), ('node summaries', self.nodes))
        return ', '.join(name for name, runs in legs if runs) or 'no outputs'


class _OutputsSupersededError(ControlFlowError):
    """The entry, its text or the job's claim changed while the outputs were generated."""


def _legs_for(source: GenerationSource) -> _Legs:
    """Choose the legs that generate the outputs the entry lacks and an active provider can produce.

    Args:
        source: The entry's text and the outputs it already holds.

    Returns:
        The legs to run. A text the synchronous write paths would not summarize gets no summary.
    """
    return _Legs(
        embedding=get_embedding_provider() is not None and not source.has_embeddings,
        summary=summary_applies(source.text_content) and not source.has_summary,
        nodes=node_layer_active() and source.node_count == 0,
    )


async def _hold_lease(repos: 'RepositoryContainer', job: ClaimedJob, *, access: BackendAccess) -> None:
    """Renew the job's lease every :data:`LEASE_RENEWAL_INTERVAL_S` seconds until the claim is lost.

    A renewal that fails is retried at the next interval; the lease then still runs.

    Args:
        repos: Repository container.
        job: The claimed job.
        access: The identity the renewals run as.
    """
    while True:
        await asyncio.sleep(LEASE_RENEWAL_INTERVAL_S)
        try:
            renewed = await repos.generation_jobs.renew_lease(job.context_id, job.claim_token, lease_s=LEASE_S, scope=access)
        except Exception as error:
            logger.warning(
                'Could not renew the lease of the generation job for context entry %s: %s',
                job.context_id, format_exception_message(error),
            )
            continue
        if not renewed:
            return


async def _generate(
    repos: 'RepositoryContainer', job: ClaimedJob, source: GenerationSource, legs: _Legs, *, access: BackendAccess,
) -> _Outputs | None:
    """Generate the outputs of the chosen legs while holding the job's lease.

    Args:
        repos: Repository container.
        job: The claimed job.
        source: The entry's text and source.
        legs: The legs to run.
        access: The identity the lease renewals run as.

    Returns:
        The embeddings, the summary and the node rows; None when the claim was lost first,
        in which case the generation was canceled. A failed embedding or summary leg
        raises its error.
    """
    generation = asyncio.create_task(
        run_generation(
            source.text_content, source.source,
            run_embedding=legs.embedding, run_summary=legs.summary, run_nodes=legs.nodes,
        ),
    )
    lease = asyncio.create_task(_hold_lease(repos, job, access=access))
    try:
        done, _pending = await asyncio.wait({generation, lease}, return_when=asyncio.FIRST_COMPLETED)
        if generation in done:
            return generation.result()
        return None
    finally:
        generation.cancel()
        lease.cancel()
        await asyncio.gather(generation, lease, return_exceptions=True)


async def _complete(
    repos: 'RepositoryContainer', job: ClaimedJob, source: GenerationSource, outputs: _Outputs, *, access: BackendAccess,
) -> None:
    """Store the outputs and delete the job in one transaction.

    The entry row is written first, so the transaction takes the entry's lock before the
    job's, in the order every write path takes them.

    Args:
        repos: Repository container.
        job: The claimed job.
        source: What the outputs were generated from.
        outputs: The embeddings, the summary and the node rows.
        access: The identity the transaction runs as.

    Raises:
        _OutputsSupersededError: If the entry no longer holds the text, or the claim no longer
            holds the job; the transaction is rolled back.
    """
    chunk_embeddings, summary, index_nodes = outputs
    async with repos.context.backend.begin_transaction(access_scope=access) as txn:
        if not await repos.context.write_generation_summary(
            job.context_id, content_hash=source.content_hash, summary=summary, txn=txn,
        ):
            raise _OutputsSupersededError('the entry was changed or deleted while its outputs were generated')
        if not await repos.generation_jobs.delete_claimed(job.context_id, job.claim_token, txn=txn):
            raise _OutputsSupersededError('the job was replaced or claimed again while its outputs were generated')
        await write_generation_outputs(
            repos, txn, job.context_id, chunk_embeddings, index_nodes,
            embedding_model=settings.embedding.model, scope=access,
        )


async def _record_failure(
    repos: 'RepositoryContainer', job: ClaimedJob, error: str, *, access: BackendAccess, max_attempts: int,
) -> JobOutcome:
    """Record a failed attempt under the job's claim and log what it did.

    Args:
        repos: Repository container.
        job: The claimed job.
        error: The failure text.
        access: The identity the write runs as.
        max_attempts: The attempts a job may spend.

    Returns:
        What the failure did to the job.
    """
    try:
        recorded = await repos.generation_jobs.record_failure(
            job.context_id, job.claim_token, error, max_attempts=max_attempts, scope=access,
        )
    except Exception as record_error:
        logger.warning(
            'Deferred generation for context entry %s failed (%s), and the failure could not be recorded (%s); '
            'the job is claimed again once its lease ends',
            job.context_id, error, format_exception_message(record_error),
        )
        return JobOutcome.UNRECORDED
    if recorded is None:
        logger.debug(
            'Deferred generation for context entry %s failed after its job was replaced or claimed again: %s',
            job.context_id, error,
        )
        return JobOutcome.DISCARDED
    if recorded.failed:
        logger.error(
            'Deferred generation for context entry %s failed after %d attempt(s): %s. The entry stays readable and '
            'FTS-searchable; restart the server or run mcp-context-server-migrate --embed-missing to retry.',
            job.context_id, recorded.attempts, error,
        )
        return JobOutcome.FAILED
    logger.warning(
        'Deferred generation for context entry %s failed on attempt %d of %d; the next attempt follows in %d s: %s',
        job.context_id, recorded.attempts, max_attempts, recorded.retry_in_s, error,
    )
    return JobOutcome.RETRYING


async def run_job(
    repos: 'RepositoryContainer', job: ClaimedJob, *, access: BackendAccess, max_attempts: int,
) -> JobOutcome:
    """Generate and store the outputs a claimed job's entry lacks.

    A job whose entry is gone is deleted. Canceling the run cancels the generation and
    records nothing; the claim stays for the caller to release or for its lease to end.

    Args:
        repos: Repository container over the backend the job runs on.
        job: The claimed job.
        access: The identity every statement runs as.
        max_attempts: The attempts a job may spend.

    Returns:
        What running the job did.
    """
    started = time.monotonic()
    try:
        source = await repos.context.read_generation_source(job.context_id, scope=access)
        if source is None:
            await repos.generation_jobs.delete_job(job.context_id, job.claim_token, scope=access)
            logger.debug('Generation job for context entry %s deleted: the entry no longer exists', job.context_id)
            return JobOutcome.DISCARDED
        legs = _legs_for(source)
        outputs: _Outputs | None = (None, None, None)
        if legs.any():
            outputs = await _generate(repos, job, source, legs, access=access)
        if outputs is None:
            logger.debug(
                'Generated outputs for context entry %s discarded: the job was replaced, reset or deleted',
                job.context_id,
            )
            return JobOutcome.DISCARDED
        await _complete(repos, job, source, outputs, access=access)
    except _OutputsSupersededError as superseded:
        logger.debug('Generated outputs for context entry %s discarded: %s', job.context_id, superseded)
        return JobOutcome.DISCARDED
    except Exception as error:
        return await _record_failure(
            repos, job, format_exception_message(error), access=access, max_attempts=max_attempts,
        )
    logger.info(
        'Deferred generation completed for context entry %s (%s) in %.1f s',
        job.context_id, legs.describe(), time.monotonic() - started,
    )
    return JobOutcome.COMPLETED

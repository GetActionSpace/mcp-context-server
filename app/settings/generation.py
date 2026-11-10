"""Deferred generation job settings."""

from pydantic import Field

from app.settings.base import CommonSettings


class GenerationJobSettings(CommonSettings):
    """Deferred generation job configuration.

    A write that defers generation stores a job that the server's worker runs later.
    Each attempt runs the embedding, summary and node-summary legs the entry still
    lacks, with the providers' own per-call retries inside it; a job that fails
    every attempt stays failed, and the entry stays readable and FTS-searchable.
    """

    max_attempts: int = Field(
        default=3,
        alias='GENERATION_JOB_MAX_ATTEMPTS',
        ge=1,
        le=10,
        description='Number of times a deferred generation job is attempted before it stays failed until the next '
                    'server start or an mcp-context-server-migrate --embed-missing or --re-embed run',
    )

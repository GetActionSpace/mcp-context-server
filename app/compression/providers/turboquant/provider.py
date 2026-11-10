"""TurboQuant :class:`CompressionProvider` implementation.

Composes the internal encoder/decoder modules and presents both sync
(hot-path) and async (event-loop-friendly) interfaces.

The async wrappers use bare :func:`asyncio.to_thread`; an outer
semaphore (``COMPRESSION_MAX_CONCURRENT``) may be layered on top by
the caller to bound CPU concurrency.
"""

import asyncio
import logging
import threading
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from app.compression.providers.turboquant import decoder
from app.compression.providers.turboquant import encoder
from app.compression.providers.turboquant._types import payload_from_bytes
from app.compression.types import CompressionMetadata
from app.settings import get_settings

logger = logging.getLogger(__name__)

# threadpoolctl's save/set/restore of the BLAS thread count is PROCESS-GLOBAL for
# the OpenBLAS that NumPy bundles: its "pthreads" and "windows" threading layers
# share one thread pool across the calling threads. The hot path fans
# encode/decode/estimate across worker threads (asyncio.to_thread bounded by
# COMPRESSION_MAX_CONCURRENT), so unsynchronized threadpool_limits blocks
# would race: one thread's exit can restore the UNPINNED count while a sibling
# still computes (breaking the pin), or capture a sibling's pinned value as the
# "original" to restore (leaking the pin after all work ends). Serializing the
# whole limit-scoped BLAS section keeps both documented contracts -- pinned
# during the GEMM, restored after -- at the cost of serializing the BLAS calls
# themselves, which is acceptable: they are the CPU-bound section the pin
# exists to keep from oversubscribing.
_blas_limits_lock = threading.Lock()


class TurboQuantProvider:
    """TurboQuant compression provider.

    Accepts explicit construction parameters; falls back to the app-wide
    settings singleton when a parameter is left at its default ``None``.
    Mixing per-parameter overrides with settings defaults is supported:
    any parameter passed explicitly takes precedence; remaining
    parameters are read from settings.

    The ``seed`` parameter is load-bearing for codebook geometry and the
    rotation matrix. The settings layer always resolves it to a concrete
    non-negative integer (default 0); production deployments validate
    seed stability at the startup-validator layer via
    ``ConfigurationError`` (exit 78) when the env value diverges from
    the persisted ``compression_metadata`` row.

    Args:
        bits: Bit-width per coordinate; supported range is [2, 4]. When
            ``None``, reads ``settings.compression.bits``.
        variant: Compression variant identifier. When ``None``, reads
            ``settings.compression.variant``.
        seed: Rotation-matrix seed. When ``None``, reads
            ``settings.compression.seed``.
        dim: Vector dimension d. When ``None``, reads
            ``settings.embedding.dim``.

    Raises:
        ValueError: If ``variant`` is not in ``{'mse', 'ip'}``.
    """

    def __init__(
        self,
        *,
        bits: int | None = None,
        variant: Literal['mse', 'ip'] | None = None,
        seed: int | None = None,
        dim: int | None = None,
    ) -> None:
        # Resolve each parameter: explicit kwarg wins; settings fallback
        # is consulted only when the kwarg is None.
        settings = get_settings()
        comp = settings.compression
        emb = settings.embedding

        resolved_bits = comp.bits if bits is None else bits
        resolved_variant_raw: str = comp.variant if variant is None else variant
        resolved_seed = comp.seed if seed is None else seed
        resolved_dim = emb.dim if dim is None else dim

        self._variant: Literal['mse', 'ip']
        if resolved_variant_raw == 'mse':
            self._variant = 'mse'
        elif resolved_variant_raw == 'ip':
            self._variant = 'ip'
        else:
            raise ValueError(
                f"variant must be 'mse' or 'ip', got {resolved_variant_raw!r}",
            )

        self._bits = resolved_bits
        self._seed = resolved_seed
        self._dim = resolved_dim

        logger.debug(
            'TurboQuantProvider initialized: bits=%d variant=%s dim=%d seed=%d',
            self._bits, self._variant, self._dim, self._seed,
        )

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return 'turboquant'

    @property
    def metadata(self) -> CompressionMetadata:
        """Provenance metadata describing the current configuration.

        ``codebook_fingerprint`` is left None here: it requires materializing the
        rotation matrix (a QR factorization), which the cheap, frequently-read
        metadata property must not force. Callers that persist provenance
        (the startup validator and the ``--compress`` CLI) compute it explicitly
        via :meth:`codebook_fingerprint`.
        """
        return CompressionMetadata(
            provider='turboquant',
            bits=self._bits,
            variant=self._variant,
            seed=self._seed,
            dim=self._dim,
        )

    def codebook_fingerprint(self) -> str:
        """Return a stable SHA-256 hex digest of the realized rotation matrix.

        The MSE codebook boundaries and the QJL sketch matrix are derived from pure,
        bit-reproducible PCG64 / closed-form computation, so the only host-dependent
        component is the ``numpy.linalg.qr`` rotation matrix (LAPACK ``geqrf``/
        ``orgqr``). Hashing its realized float32 bytes for this ``(dim, seed)`` yields
        a digest that is identical across runs on the same numerical stack and
        DIFFERS when a different BLAS/LAPACK build or CPU dispatch produces a divergent
        rotation -- the exact cross-host corruption the startup validator must catch.

        Returns:
            Lowercase hex SHA-256 digest of the realized rotation matrix.
        """
        import hashlib

        from app.compression.providers.turboquant._rotation import get_cached_rotation

        rotation = get_cached_rotation(dim=self._dim, seed=self._seed)
        return hashlib.sha256(rotation.matrix.tobytes()).hexdigest()

    def encode_sync(self, vectors: NDArray[np.float32]) -> bytes:
        """Encode vectors and return a serialized payload.

        Pins BLAS threads to 2 for the GEMM call so that concurrent
        encodes (bounded by ``COMPRESSION_MAX_CONCURRENT`` in production
        and by CLI batch execution in :mod:`app.cli.migrate_compression.compress_execution`)
        do not oversubscribe CPU on multi-core hosts. The
        ``with threadpool_limits(...)`` block is scoped to this method;
        upstream BLAS thread counts are restored on exit.

        Args:
            vectors: Float32 array of shape ``(n, d)``.

        Returns:
            Serialized payload bytes.
        """
        with _blas_limits_lock, threadpool_limits(limits=2, user_api='blas'):
            payload = encoder.encode(
                vectors,
                bits=self._bits,
                variant=self._variant,
                seed=self._seed,
            )
        return payload.to_bytes()

    def decode_sync(self, payload: bytes) -> NDArray[np.float32]:
        """Deserialize a payload and reconstruct vectors.

        Pins BLAS threads to 2 so concurrent decodes do not oversubscribe
        CPU; same rationale as :meth:`encode_sync`.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.

        Returns:
            Reconstructed float32 vectors.
        """
        decoded_payload = payload_from_bytes(payload)
        with _blas_limits_lock, threadpool_limits(limits=2, user_api='blas'):
            return decoder.decode(decoded_payload)

    def estimate_inner_product_sync(
        self,
        payload: bytes,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Estimate inner products between DB rows and queries.

        Pins BLAS threads to 2 so concurrent estimates do not
        oversubscribe CPU; same rationale as :meth:`encode_sync`. The
        thread pin applies inside the worker thread when this method is
        invoked from :meth:`estimate_inner_product` via
        :func:`asyncio.to_thread`.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.
            queries: Query matrix of shape ``(nq, d)``.

        Returns:
            Float32 array of shape ``(nq, n_rows)``.
        """
        decoded_payload = payload_from_bytes(payload)
        with _blas_limits_lock, threadpool_limits(limits=2, user_api='blas'):
            return decoder.estimate_inner_product(decoded_payload, queries)

    def warmup(self) -> None:
        """Prime NumPy lazy imports, the rotation cache, and the threadpoolctl probe.

        ``store_context`` fans per-chunk encodes across worker threads via
        :func:`asyncio.to_thread`. On a cold process that first concurrent batch
        would trigger NumPy's lazy submodule import (``numpy.random`` /
        ``numpy.linalg``, reached through rotation-matrix generation) inside several
        threads at once, deadlocking against :mod:`threadpoolctl`'s
        ``dl_iterate_phdr`` (the dynamic-loader lock taken under the Python import
        lock). Running one full round-trip single-threaded here populates the
        ``(dim, seed)`` rotation cache and completes every lazy import before any
        concurrency, so the later threaded encodes only hit warm caches. The probe
        is best-effort: a failure is logged and swallowed because the goal is
        priming, not validation -- a genuine encode error still surfaces on the
        real call.
        """
        probe = np.ones((1, self._dim), dtype=np.float32)
        try:
            payload = self.encode_sync(probe)
            self.decode_sync(payload)
            self.estimate_inner_product_sync(payload, probe)
        except Exception as exc:
            logger.warning(
                'Compression warmup round-trip failed (priming may be partial): %s',
                exc,
            )

    async def encode(self, vectors: NDArray[np.float32]) -> bytes:
        """Async wrapper around :meth:`encode_sync` via :func:`asyncio.to_thread`.

        Args:
            vectors: Float32 array of shape ``(n, d)``.

        Returns:
            Serialized payload bytes.
        """
        return await asyncio.to_thread(self.encode_sync, vectors)

    async def decode(self, payload: bytes) -> NDArray[np.float32]:
        """Async wrapper around :meth:`decode_sync`.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.

        Returns:
            Reconstructed float32 vectors.
        """
        return await asyncio.to_thread(self.decode_sync, payload)

    async def estimate_inner_product(
        self,
        payload: bytes,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Async wrapper around :meth:`estimate_inner_product_sync`.

        Offloads the GEMM into a worker thread via
        :func:`asyncio.to_thread` so callers inside an active event loop
        (notably :meth:`EmbeddingRepository.search_compressed`) do not
        block other concurrent MCP requests. The BLAS thread pin lives
        inside :meth:`estimate_inner_product_sync` itself, so this
        wrapper inherits the same protection without duplicating the pin.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.
            queries: Query matrix of shape ``(nq, d)``.

        Returns:
            Float32 array of shape ``(nq, n_rows)``.
        """
        return await asyncio.to_thread(self.estimate_inner_product_sync, payload, queries)

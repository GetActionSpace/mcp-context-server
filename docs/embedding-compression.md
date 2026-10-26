# Embedding Compression Guide

Embedding compression replaces the fp32 chunk-embedding storage with bit-packed compressed payloads, reducing semantic-search storage by approximately 8x at the default configuration. The feature is **on by default** in v3.0.0; set `ENABLE_EMBEDDING_COMPRESSION=false` to opt out and keep fp32 storage (on a database that already holds compressed data, decode it back to fp32 with `mcp-context-server-migrate --decompress` first -- a bare disable is refused at startup with exit 78; see [Reversing the compression](#reversing-the-compression)). Fresh installations get compressed storage automatically; existing installations with populated fp32 embeddings MUST run the `mcp-context-server-migrate --compress` CLI to convert them in place -- enabling compression by environment variable alone on such a database is refused at startup with exit 78, because the schema swap would otherwise drop the fp32 embeddings permanently.

This guide explains what the feature does, how to configure it, how the data is stored, what trade-offs apply, and how to migrate an existing database.

## Overview

When `ENABLE_EMBEDDING_COMPRESSION=true`, the storage write path encodes every fp32 embedding through the TurboQuant compression provider and persists the bit-packed payload instead of the original float vector. The semantic and hybrid search read paths route through a compressed scorer that estimates the original similarity directly from the compressed payload, with no intermediate decode step on the hot path.

Use embedding compression when chunk-embedding storage cost is becoming the limiting factor of your deployment. At d=1024 and bits=4 (the defaults), each chunk uses roughly 512 bytes of payload instead of 4 kilobytes, while top-K recall stays at parity with the fp32 baseline on realistic corpora (measured top-K overlap of 1.0 across the recall-regression test cells; the project's regression gate requires at least 0.85).

Key properties:

- Default state is **on** in v3.0.0 with `COMPRESSION_SEED=0`. Set `ENABLE_EMBEDDING_COMPRESSION=false` to opt out and use fp32 storage; on a database that already holds compressed data you must run `mcp-context-server-migrate --decompress` BEFORE disabling (a bare env flip is refused at startup with exit 78, because the compressed embeddings would otherwise become invisible to search). The enable direction is guarded symmetrically: on a database that still holds fp32 embeddings and has never been compressed (no provenance row), enabling compression by environment variable alone is ALSO refused at startup with exit 78, because the compression migration's table swap would otherwise drop those embeddings permanently — run `mcp-context-server-migrate --compress` to encode them first (see [Migration CLI](#migration-cli)).
- Compression provisioning follows embedding storage: with `ENABLE_EMBEDDING_GENERATION=false` on a FRESH database -- no compression schema and no embedding infrastructure -- the server creates neither the compressed table nor a provenance row (nothing can ever write a compressed payload on that configuration), so a later `ENABLE_EMBEDDING_COMPRESSION=false` flip has nothing to wedge on. A database that already carries the compression schema -- data compressed while generation was on -- keeps being maintained and seed-validated so the decode path stays protected, and a database whose embedding tables were ever provisioned (an `embedding_metadata` table exists) still receives the compression schema, because the delete/update cleanup paths route through the compressed payload table whenever compression is enabled. The fp32 side follows the SAME rule symmetrically: with compression off, the semantic and chunking migrations re-provision the fp32 layout on an infra-carrying database even while generation is toggled off, so a compression-off flip (or a zero-data `--decompress`) on such a database self-heals at the next startup instead of stranding the cleanup paths on a missing table. On PostgreSQL every provisioning probe resolves table names through the connection `search_path` (`to_regclass`), exactly like the bare-name DDL and the runtime cleanup gates it protects, so a legacy table living in `public` is seen by the guard that protects it even under a non-default `POSTGRESQL_SCHEMA`. The enable-direction guard above applies REGARDLESS of the generation toggle: a populated fp32 table with no provenance row refuses to start (exit 78) either way.
- Storage layout is **replacement**, not parallel: the fp32 `vec_context_embeddings` table is dropped when compression is enabled, and the compressed `vec_context_embeddings_compressed` table takes its place. A singleton `compression_metadata` row records the provenance.
- The configuration is **seed-locked** at first startup. The rotation seed is recorded in the database and may not be changed without re-encoding every payload.
- The MCP tool surface is **unchanged**. `semantic_search_context`, `hybrid_search_context`, `store_context`, and the batch tools work identically; only the underlying byte layout differs.

## Architecture

The compressed subsystem reuses the existing chunking pipeline and storage backends; only the bytes inside the embedding tables change.

### Replacement storage

When compression is enabled, the embedding write path stores one bit-packed payload per chunk in `vec_context_embeddings_compressed`. The legacy fp32 `vec_context_embeddings` table is removed by the schema migration and is not maintained in parallel. The compressed write path does NOT maintain the `embedding_chunks` bridge (that table backs only the fp32 read path's INTEGER `vec_rowid` join); the 1:N chunk-to-context mapping is carried by `vec_context_embeddings_compressed` itself — one row per chunk, keyed by `context_id` with a per-context sequential `chunk_index`, the outer `context_id` foreign key referencing `context_entries.id`.

`--decompress` reverses compression WITHOUT depending on any pre-existing `embedding_chunks` rows: the reverse loop in `app/cli/migrate_compression/decompress_execution.py` decodes each compressed payload and rebuilds BOTH the recreated fp32 `vec_context_embeddings` table and the `embedding_chunks` bridge directly from each compressed row's own `context_id`/`start_index`/`end_index`, linking them through a fresh `vec_rowid`. This works identically whether the compressed rows were produced by the live server (which never writes `embedding_chunks`) or by the CLI `--compress` migration, which stores `chunk_index` with the same per-context sequential semantics the live write path uses.

The compressed table schema:

| Column         | SQLite type | PostgreSQL type | Notes                                                   |
|----------------|-------------|-----------------|---------------------------------------------------------|
| `id`           | INTEGER     | BIGSERIAL       | Internal rowid bridge (not exposed at the MCP boundary) |
| `context_id`   | TEXT        | UUID            | FK -> `context_entries(id)` (32-char hex / native UUID) |
| `chunk_index`  | INTEGER     | INTEGER         | Zero-based chunk position                               |
| `start_index`  | INTEGER     | INTEGER         | Character start offset within `text_content`            |
| `end_index`    | INTEGER     | INTEGER         | Character end offset within `text_content`              |
| `payload`      | BLOB        | BYTEA           | Bit-packed compressed payload                           |
| `created_at`   | TEXT        | TIMESTAMPTZ     | Row creation timestamp                                  |

The `payload` BYTEA/BLOB has no fixed width; its size depends on `dim`, `bits`, and `variant` (see [Storage Math](#storage-math)).

The compressed write path obtains its TurboQuant provider from `get_cached_compression_provider()` in `app/compression/factory.py`. The cache is process-wide; a `reset_cached_compression_provider()` hook is exposed from the same module so administrative tooling and test fixtures can force fresh construction after configuration changes. The embedding repository's `_reset_compression_cache()` helper delegates to this factory reset alongside the metadata cache invalidation it already performs, so a single call clears every cached compression component.

### Singleton provenance: `compression_metadata`

A single-row provenance table records the configuration that was active when the database was first compressed:

```sql
CREATE TABLE compression_metadata (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    provider              TEXT NOT NULL,
    bits                  INTEGER NOT NULL,
    variant               TEXT NOT NULL,
    seed                  INTEGER NOT NULL,
    dim                   INTEGER NOT NULL,
    codebook_fingerprint  TEXT,
    created_at            TEXT NOT NULL
)
```

The `CHECK (id = 1)` constraint enforces the singleton property at the SQL layer. The bootstrap-only startup validator reads this row on every start, compares it to the active `CompressionSettings`, and refuses to start when the runtime values diverge from the recorded values. This is the seed-locked invariant: once data has been encoded with a given seed, no other seed can decode it correctly.

`codebook_fingerprint` is a lowercase hex SHA-256 of the REALIZED rotation matrix (`numpy.linalg.qr` output for `(dim, seed)`), recorded at first compression. The seed alone does NOT guarantee a reproducible codebook: `numpy.linalg.qr` is a LAPACK `geqrf`/`orgqr` call whose low-order bits differ across BLAS/LAPACK builds and CPU dispatch, so the same `(dim, seed)` can materialize a different rotation on a different host. The validator re-derives this digest on every start (when the scalar fields match) and raises `ConfigurationError` (exit 78) on divergence, converting an otherwise-silent cross-host corruption of every decode/search into a loud startup failure. The column is nullable so a row written before fingerprinting still reads (the warning notes that cross-host divergence cannot then be detected for that database).

### Seed-locked invariant

The TurboQuant rotation matrix is deterministic given a seed AND a fixed numerical stack. Two pods with different seeds will produce incompatible encodings: payloads written by one pod will not decode correctly with the other pod's rotation. To prevent silent corruption, the validator treats the seed as load-bearing and refuses to start when the runtime value disagrees with the recorded value. Because the realized rotation also depends on the host's BLAS/LAPACK build and CPU (numpy.linalg.qr is not bit-reproducible across hosts even for a fixed seed), the validator additionally re-derives the `codebook_fingerprint` and refuses to start when the realized codebook diverges from the recorded one.

In multi-pod deployments (Kubernetes, multiple horizontal-scale workers, multiple processes against the same database), every pod MUST inherit the same `COMPRESSION_SEED`. The recommended pattern is a ConfigMap-bound env var; see [Multi-Pod Kubernetes Deployments](#multi-pod-kubernetes-deployments) below.

### Algorithms

The provider implements two variants from the TurboQuant paper (arXiv 2504.19874v1):

- `mse` (Algorithm 1): an L2-optimal scalar quantizer. Decoding reconstructs an approximation of the original vector; similarity scoring decodes the payload and computes L2 distance against the query.
- `ip` (Algorithm 2 with QJL): an unbiased inner-product estimator that reserves one bit per coordinate for the Johnson-Lindenstrauss-style sign and uses the remaining bits for an MSE-style component. The provider can estimate the inner product directly from the packed payload without an explicit decode step.

For similarity search over normalized embeddings (the typical case), `ip` is the recommended variant and is the default. For workloads that require L2 reconstruction quality (for example, downstream pipelines that consume the decoded vector), `mse` is preferable.

### Compressed payload wire format

Each row of `vec_context_embeddings_compressed` stores a single bit-packed payload as a length-self-describing blob. The variant is encoded in the blob itself, so the storage layer never needs an out-of-band tag.

Two concrete dataclasses model the on-disk shape:

- `MSEPayload` -- fields `bits`, `dim`, `seed`, `n_rows`, `norms`, `packed_indices`. Header layout `<4sBBHII` (4-byte magic prefix, variant code byte `0`, `bits` byte, `dim` as `H`, `n_rows` as `I`, `seed` as `I`) followed by length-prefixed `norms` and `packed_indices` blocks.
- `IPPayload` -- fields `bits`, `mse_bits`, `dim`, `seed`, `n_rows`, `norms`, `residual_norms`, `qjl_bits`, `packed_indices`. Header layout `<4sBBHIIB3x` (the MSE header plus a single `mse_bits` byte and three padding bytes for 4-byte alignment of the following length-prefixed blocks) followed by length-prefixed `norms`, `residual_norms`, `qjl_bits`, and `packed_indices` blocks.

A module-level dispatcher `payload_from_bytes(blob)` reads the 1-byte variant code at offset 4 and delegates to `MSEPayload.from_bytes` (code `0`) or `IPPayload.from_bytes` (code `1`); unknown magic or unknown variant codes raise `ValueError`. Decoders that have already typed the payload object can `match` on the concrete subtype directly.

The explicit `mse_bits` byte stored on every IP payload is what lets the decoder reconstruct the inner-MSE quantizer without assuming any particular relationship between the QJL bit width and the outer `bits` value. The current encoder writes `mse_bits = bits - 1` (one bit per coordinate is reserved for the QJL sign), but the wire format does not bake that convention into the decoder: any future encoder that uses a different inner-MSE bit width can be decoded correctly because the value is read straight out of the IP header. The algorithm itself is unchanged from arXiv 2504.19874v1; only the on-disk metadata is more self-describing.

## Configuration

Six environment variables control the feature. Defaults are chosen so that compression is on out of the box with sensible recall/storage trade-offs; operators who want fp32 storage opt out by setting `ENABLE_EMBEDDING_COMPRESSION=false` (decode an already-compressed database back to fp32 with `--decompress` first; a bare disable is refused at startup -- see [Reversing the compression](#reversing-the-compression)).

| Variable                       | Default             | Range / Values             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|--------------------------------|---------------------|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_EMBEDDING_COMPRESSION` | `true`              | boolean                    | Enable TurboQuant embedding compression at storage time. Default `true` in v3.0.0. Set to `false` to opt out and keep fp32 storage. When `true`, fp32 embeddings are replaced with bit-packed compressed payloads (~8x storage reduction at default `bits=4`). Disabling on a database that already holds compressed data is refused at startup (exit 78) until the embeddings are decoded back to fp32 with `mcp-context-server-migrate --decompress`. |
| `COMPRESSION_PROVIDER`         | `turboquant`        | `turboquant`               | Compression provider. Only `turboquant` is supported in v3.0.0.                                                                                                                                                                                                                                                                                                                                                                                         |
| `COMPRESSION_BITS`             | `4`                 | integer in [2, 4]          | Bits per coordinate. `2` = 16x storage savings, `3` = ~11x, `4` = 8x. The lower bound of 2 is required by `variant='ip'`, which reserves one bit per coordinate for the QJL sign.                                                                                                                                                                                                                                                                       |
| `COMPRESSION_VARIANT`          | `ip`                | `ip` or `mse`              | `ip` (default): Algorithm 2 with QJL, unbiased inner-product estimator. `mse`: Algorithm 1, L2-optimal reconstruction.                                                                                                                                                                                                                                                                                                                                  |
| `COMPRESSION_SEED`             | `0`                 | integer in [0, 4294967295] | Rotation matrix seed. Default `0`. Pick any stable integer in `[0, 4294967295]` (the seed is packed into the compressed payload as an unsigned 32-bit field); the value is persisted in `compression_metadata` at first startup and treated as load-bearing thereafter. Mismatches at later startups raise `ConfigurationError` (exit 78) and refuse the server.                                                                                        |
| `COMPRESSION_MAX_CONCURRENT`   | `min(cpu_count, 4)` | integer in [1, 32]         | Max concurrent compression encode workers dispatched to threads; the CPU-bound codec section is serialized by a process-wide BLAS-limits lock, so this bounds worker fan-out, not CPU parallelism. Separate from the I/O-bound `EMBEDDING_MAX_CONCURRENT` and `SUMMARY_MAX_CONCURRENT` semaphores.                                                                                                                                                      |

`COMPRESSION_SEED` is bootstrap-locked. On the first start, the validator writes the active `(provider, bits, variant, seed, dim)` tuple to `compression_metadata`. On every subsequent start, the validator reads the row and compares each field; any mismatch raises `ConfigurationError` (exit 78) and the supervisor will not auto-restart. This loud failure mode is intentional: it surfaces seed drift before silently corrupted data accumulates. The default `0` is statistically equivalent to any other integer in `[0, 4294967295]` (the wire format packs the seed as an unsigned 32-bit field, so values outside that range are rejected at startup); override it only if you have an operational reason to choose a different seed before any compressed data is written.

See [Environment Variables Reference](environment-variables.md#embedding-compression-settings) for the canonical list of compression settings and their constraint metadata.

### Provider construction contract

`TurboQuantProvider` exposes a keyword-only constructor with explicit `bits`, `variant`, `seed`, and `dim` parameters. Each parameter defaults to `None`; when omitted, the value falls back to the corresponding setting resolved through `get_settings()` -- `compression.bits`, `compression.variant`, `compression.seed`, and `embedding.dim` respectively. This explicit-kwarg surface is what the migration CLI uses to instantiate a provider against a persisted `compression_metadata` row without mutating process environment variables.

Because `compression.seed` is typed as `int` with default `0` (v3.0.0), every kwarg slot has a deterministic resolution path: explicit `seed=` wins, otherwise the env var (`COMPRESSION_SEED`) wins, otherwise the default `0`. The provider is therefore never asked to build a rotation with an indeterminate seed, and payload provenance is unambiguous from the first encode without any opt-in operator action.

### Quick start

For a fresh installation:

1. Install the server normally. No additional Python extras are required for compression; the TurboQuant provider uses only NumPy, which is a core dependency in v3.0.0.
2. (Optional) Override the compression env vars only if you need non-default behavior:

   ```bash
   # All four are optional in v3.0.0; the values below are the defaults.
   # Override only if you intentionally want a different configuration.
   # export ENABLE_EMBEDDING_COMPRESSION=true
   # export COMPRESSION_BITS=4
   # export COMPRESSION_VARIANT=ip
   # export COMPRESSION_SEED=0
   ```

3. Start the server. The migration loader creates `vec_context_embeddings_compressed` and `compression_metadata`; the validator inserts the singleton row with the active seed (default `0`); subsequent stores use the compressed write path.

For an existing installation with fp32 data, follow the [Migration CLI](#migration-cli) section below before changing the env vars.

## Observability

Four observation surfaces let operators verify the active compression configuration and measure embedding storage without inspecting the database directly.

### Startup log line

Immediately after the bootstrap validator confirms the persisted `compression_metadata` row matches the runtime `CompressionSettings`, the server lifespan emits a single INFO log line announcing the active configuration. When `ENABLE_EMBEDDING_COMPRESSION=true` and a provenance row exists, the message takes the form:

```text
Embedding compression enabled with provider: turboquant (bits=4, variant=ip, dim=1024, seed=0, max_concurrent=4)
```

The values for `provider`, `bits`, `variant`, `dim`, and `seed` are read from the singleton `compression_metadata` row (database-of-record), so the line reflects what the data was encoded with rather than the raw env vars. `max_concurrent` comes from the runtime `CompressionSettings` because the semaphore is a process-local resource, not a property of the stored payloads. When `ENABLE_EMBEDDING_COMPRESSION=false`, the lifespan instead emits:

```text
Embedding compression disabled (ENABLE_EMBEDDING_COMPRESSION=false)
```

The line is emitted at INFO level, sibling to the existing announcements for embedding generation, reranking, chunking, and summary; enable INFO logging (`LOG_LEVEL=INFO`) to surface it at startup.

### `get_statistics` `compression` block

The `get_statistics` MCP tool returns a top-level `compression` sub-block alongside the other feature blocks (semantic_search, fts, chunking, reranking, summary). When `ENABLE_EMBEDDING_COMPRESSION=true` and the provenance row has been bootstrapped, the block has the shape:

```json
{
  "enabled": true,
  "available": true,
  "provider": "turboquant",
  "bits": 4,
  "variant": "ip",
  "seed": 0,
  "dim": 1024,
  "max_concurrent": 4
}
```

`provider`, `bits`, `variant`, `seed`, and `dim` come from the singleton `compression_metadata` row; `max_concurrent` comes from the runtime `CompressionSettings`. When `ENABLE_EMBEDDING_COMPRESSION=false`, the block collapses to `{"enabled": false, "available": false}`. When compression is enabled but the provenance row has not yet been bootstrapped (rare; the validator normally raises before this point), the block reports `{"enabled": true, "available": false, "message": "Compression enabled but provenance not bootstrapped"}`.

### `compression_metadata` provenance row

The authoritative record of the active compression configuration lives in the singleton `compression_metadata` table (one row, `id = 1`). The startup log line and the `get_statistics` `compression` block both source their `provider`, `bits`, `variant`, `seed`, and `dim` values from this row. Operators who need to inspect the persisted provenance directly can read it with a single SQL query against the configured database:

```sql
SELECT provider, bits, variant, seed, dim, created_at
FROM compression_metadata
WHERE id = 1;
```

The same row drives the seed-locked invariant: every subsequent startup compares the runtime `CompressionSettings` against these persisted values and refuses to start when they disagree. See [Singleton provenance: `compression_metadata`](#singleton-provenance-compression_metadata) for the schema definition and the seed-locking semantics.

### Embedding storage size in `get_statistics`

The `get_statistics` MCP tool reports the storage occupied by embedding vectors via two top-level keys, `embeddings_size_mb` and `embeddings_size_estimated`, displayed immediately after the total `database_size_mb`. They are gated on embedding generation OR compression being enabled (`ENABLE_EMBEDDING_GENERATION=true` or `ENABLE_EMBEDDING_COMPRESSION=true`), NOT on `ENABLE_SEMANTIC_SEARCH`, so the size still surfaces in compression-on / semantic-search-off deployments. The figure covers only the active vector payload table: `vec_context_embeddings_compressed` when compression is enabled, otherwise the fp32 `vec_context_embeddings`. Computation degrades to `0.0` (with a warning) if the active table is missing, so a failure here never breaks the rest of the statistics response.

The value is NOT byte-comparable across backends:

- **PostgreSQL** reports the on-disk relation size INCLUDING indexes, via `pg_total_relation_size(to_regclass(...))`. The `to_regclass` call NULL-guards a dropped table (the compression migration drops `vec_context_embeddings` on PostgreSQL), so a missing table yields `0` rather than an `UndefinedTableError`. `embeddings_size_estimated` is always `false` on PostgreSQL.
- **SQLite, compression enabled** reports the exact raw compressed payload bytes, `SUM(LENGTH(payload))` over `vec_context_embeddings_compressed`. The computation is intentionally `dbstat`-free (the bundled SQLite build does not compile the `dbstat` virtual table). `embeddings_size_estimated` is `false`.
- **SQLite, compression disabled** reports a deterministic fp32 estimate, `SUM(chunk_count * dimensions * 4)` over `embedding_metadata` (four bytes per float). This is the only case where `embeddings_size_estimated` is `true`.

The companion `database_size_mb` key is likewise per-backend: on PostgreSQL it is the whole database via `pg_database_size(current_database())` (the local `DB_PATH` is irrelevant to a remote database and is never file-stat'd); on SQLite it is the on-disk database file size, which excludes the `-wal`/`-shm` sidecars and can therefore transiently under-report under WAL mode. The key is omitted for in-memory or missing-file SQLite databases.

## Variant Matrix

The two variants trade off reconstruction quality versus inner-product fidelity. Both round-trip within the distortion bound predicted by the paper.

| Variant | Algorithm                           | Best for                                        | Storage at d=1024 b=4 |
|---------|-------------------------------------|-------------------------------------------------|-----------------------|
| `ip`    | Algorithm 2 with QJL (default)      | Similarity search; cosine / dot-product queries | 512 bytes/chunk       |
| `mse`   | Algorithm 1 (L2-optimal quantizer)  | L2 reconstruction; downstream decode pipelines  | 512 bytes/chunk       |

`ip` is the recommended choice for typical retrieval workloads where queries are normalized embeddings and the ranking signal is cosine similarity. The QJL transform reserves one bit per coordinate for the sign, which is why `variant='ip'` requires at least 2 bits per coordinate.

`mse` is appropriate when you need to decode the compressed payload back to an approximate fp32 vector and feed it to a downstream consumer that does its own scoring. The decoded vector is bounded by the Theorem 1 distortion bound from the paper.

The on-disk variant code byte (`0` = MSE, `1` = IP) is what the `payload_from_bytes` dispatcher reads to select the right subtype when decoding; see [Compressed payload wire format](#compressed-payload-wire-format) for the per-variant header layout.

## Storage Math

The byte-per-chunk formulas:

- `variant='mse'`: `bytes_per_chunk = ceil(dim * bits / 8)`
- `variant='ip'`:  `bytes_per_chunk = ceil(dim * (bits - 1) / 8) + ceil(dim / 8)` (MSE component on `bits - 1` bits, plus the QJL sign on 1 bit per coordinate)

Worked example at d=1024, b=4, variant='ip':

- MSE component: `ceil(1024 * 3 / 8) = 384` bytes
- QJL sign bits: `ceil(1024 / 8) = 128` bytes
- Total: **512 bytes/chunk** vs **4096 bytes/chunk** for fp32 -> **~8x compression**

Bulk example at the same configuration:

| Chunks    | fp32 storage | Compressed storage | Reduction |
|-----------|--------------|--------------------|-----------|
| 100 000   | ~410 MB      | ~52 MB             | ~8x       |
| 1 000 000 | ~4.1 GB      | ~520 MB            | ~8x       |
| 10 000 000| ~41 GB       | ~5.2 GB            | ~8x       |

These numbers exclude per-row overhead (rowid, foreign key, chunk indices, timestamps), which is identical between the fp32 and compressed paths.

## Performance Profile

### Per-vector encode latency

Encoding a single vector at d=1024, bits=4, variant='ip' on a single CPU thread takes approximately **445-477 microseconds** measured across runs. This is the steady-state cost once the rotation and QJL caches are warm; the first encode per process pays a one-time cost for cache initialization.

The encode happens inside the storage write path. The `COMPRESSION_MAX_CONCURRENT` semaphore is acquired ONCE PER ENCODE inside the per-chunk `_encode_one` coroutine, so each in-flight encode holds at most one permit, and an N-chunk `store_context` request dispatches at most `COMPRESSION_MAX_CONCURRENT` encode workers to threads at any moment regardless of how many chunks the request contains. The semaphore bounds worker FAN-OUT (queued `asyncio.to_thread` threads and their memory), not CPU parallelism: the codec body itself runs under a process-wide BLAS-limits lock (next paragraph), so at most one encode, decode, or estimate executes its CPU-bound section at a time and raising `COMPRESSION_MAX_CONCURRENT` does not increase encode throughput.

Each invocation of `TurboQuantProvider.encode_sync`, `decode_sync`, and `estimate_inner_product_sync` runs inside `with _blas_limits_lock, threadpool_limits(limits=2, user_api='blas'):` (from the `threadpoolctl` core runtime dependency). The pin bounds BLAS thread usage per invocation so a codec call does not oversubscribe CPU on multi-core hosts where BLAS would otherwise spawn one worker thread per logical core per call; the module-level `_blas_limits_lock` serializes the pinned bodies because `threadpoolctl`'s save/set/restore of BLAS thread counts is process-global, and concurrent pins would race each other's restore. Upstream BLAS thread counts are restored on exit. Because the lock and pin live inside the synchronous methods themselves, every caller (the async `encode`/`decode`/`estimate_inner_product` wrappers used by the server, the `_encode_one` coroutine in the storage write path, and the migration CLI's batched encode loop) inherits both without duplicating the wrapper.

### Linear-scan search scale

The compressed read path performs a memory-bounded linear scan over the candidate set produced by metadata, thread, tag, and date filters. The scorer decodes every candidate payload via `payload_from_bytes`, calls the discriminated subtype's `concat` classmethod (`MSEPayload.concat` or `IPPayload.concat`) to combine all same-variant payloads into ONE synthetic payload, and invokes the TurboQuant provider exactly once per query batch: `await provider.estimate_inner_product(...)` for `variant='ip'`, `await provider.decode(...)` for `variant='mse'`. The async wrappers offload the underlying GEMM into a worker thread via `asyncio.to_thread(...)` so the event loop stays responsive to other concurrent MCP requests during the scan; the synchronous variants (`estimate_inner_product_sync`, `decode_sync`) remain available for non-async callers such as the migration CLI. Per-context aggregation, pagination, and hydration run over the single combined result and are otherwise unchanged.

Because the provider is invoked once instead of once per candidate row, latency scales with the underlying GEMM cost (the size of the matrix-matrix multiply performed by the variant's inner kernel) rather than with the number of per-row provider calls. Sub-second query times are comfortable up to approximately 100 000 to 1 000 000 compressed chunks per thread. Beyond that, the linear scan itself dominates latency.

To keep queries fast at scale:

- Use selective filters (`thread_id`, `tags`, `metadata`) so the candidate set is bounded before scoring runs.
- Keep the average chunk count per query reasonable; very large unfiltered corpora are best served by approximate nearest-neighbor (ANN) indexing, which is planned for a future release (see [Future Capabilities](#future-capabilities)).

### Recall

The recall regression suite asserts that top-K overlap with fp32 ground truth is at least 0.85 separately for the cells `(bits=2, ip)`, `(bits=2, mse)`, `(bits=3, ip)`, and `(bits=4, ip)` on a deterministic synthetic d=1024 corpus. Measured recall in the project's test suite is 1.0 at all four cells with margin. Batched scoring is numerically equivalent to per-row scoring within float32 precision (BLAS reduction-order differences across batch sizes are bounded; the regression suite confirms 1.000 overlap across the supported `(bits, variant)` cells).

## Multi-Pod Kubernetes Deployments

When multiple pods (or multiple processes) share a single database, all of them must encode and decode with the same rotation. This means every pod must inherit the same `COMPRESSION_SEED`. A mismatched seed in any pod will silently produce payloads that no other pod can decode correctly.

The validator catches the cross-pod mismatch at startup: each pod compares its runtime seed against the persisted `compression_metadata.seed` value, and any disagreement raises `ConfigurationError` (exit 78). Kubernetes will not auto-restart on exit 78, which surfaces the misconfiguration loudly instead of letting it manifest as silently corrupted search results.

Recommended pattern:

- Define `COMPRESSION_SEED` once in a Helm value or ConfigMap.
- Reference it from the Pod env via the ConfigMap, not as a literal in the Pod spec.
- Treat the value as immutable for the lifetime of the database.

The Helm chart ships an active compression block in both `values-sqlite.yaml` and `values-postgresql.yaml` (`enabled: true`, `seed: 0` in v3.0.0). Set `compression.enabled: false` in your Helm values to opt out of compression on a given deployment. The PostgreSQL profile additionally documents the multi-pod ConfigMap discipline inline, including the requirement that every pod inherit the same seed via the chart's ConfigMap template.

## PostgreSQL `search_path` and `POSTGRESQL_SCHEMA` Contract

The compression migration, like the other PostgreSQL migrations (`add_semantic_search_postgresql.sql`, `add_chunking_postgresql.sql`, `add_fts_postgresql.sql`), uses BARE table names in TABLE and INDEX DDL. TABLE/INDEX resolution relies on PostgreSQL's `search_path` rather than explicit schema qualification. This matches the convention established by `app/schemas/postgresql_schema.sql` (the main schema) and `app/repositories/embedding_repository/` (the read path), so the migration loader, the migration CLI (`mcp-context-server-migrate`), and the runtime application all create, write, and read tables in the same schema.

**Operator contract:** When `POSTGRESQL_SCHEMA` is set to any value other than the default `public`, you MUST ensure that `search_path` includes `$POSTGRESQL_SCHEMA` as the first element on every connection used by:

- The MCP server's connection pool
- The migration CLI (`mcp-context-server-migrate --compress`, `--decompress`)
- Any external maintenance tool or admin session that issues DDL (for example, manual `psql` invocations)

The MCP server enforces this contract automatically: every connection it opens runs `SET search_path = "<schema>", public` before its first statement, where `<schema>` is the resolved `POSTGRESQL_SCHEMA` value, double-quoted so mixed-case and reserved identifiers are handled safely. Pooled connections get it from the pool's setup callback, which runs on every acquire and therefore also restores it after the pool's `RESET ALL` on release. It is deliberately NOT a startup-packet parameter, because an external pooler refuses any startup parameter it does not allowlist (see [Startup Parameters and Pooler Allowlists](database-backends.md#startup-parameters-and-pooler-allowlists)). When `POSTGRESQL_SCHEMA=public` (the default), the statement sets `search_path = "public", public`, a benign no-op matching PostgreSQL's built-in default.

The migration CLI (`mcp-context-server-migrate --compress`, `--decompress`, `--embed-missing`) applies the SAME statement on each connection it dials, so it resolves the configured schema exactly as the server does and needs no operator action either. Only EXTERNAL maintenance callsites -- manual `psql` invocations, admin sessions, third-party tooling -- have to set the search path themselves, via a per-connection `SET` or the connection string's `options` parameter (`?options=-csearch_path%3D$POSTGRESQL_SCHEMA%2Cpublic`). When `POSTGRESQL_SCHEMA=public`, no action is required for those either.

`FUNCTION` DDL remains schema-qualified (`{SCHEMA}.update_updated_at_column`, `{SCHEMA}.update_embedding_metadata_timestamp`, `{SCHEMA}.jsonb_merge_patch`) as a deliberate CVE-2018-1058 mitigation that prevents `search_path` hijacking of trigger functions. This contract is independent of the table-name resolution above: functions are resolved by explicit schema qualification at every call site (triggers, recursive function calls, ALTER FUNCTION targets).

Idempotency-check filters inside migrations use `current_schema()` (rather than a templated schema name) so they introspect whatever schema `search_path` resolves to. This means correctly-configured operators see migrations as idempotent under both default and non-default `POSTGRESQL_SCHEMA` values; misconfigured operators (with `search_path` not aligned to `$POSTGRESQL_SCHEMA`) get a visibly inconsistent result rather than silent corruption.

The project's PostgreSQL integration test suite enforces this contract as a regression gate. A dedicated `pg_non_default_schema_db` fixture provisions an isolated database with a non-default `mcp_test` schema pre-created, and the `tests/integration/postgresql/non_default_schema/` suite runs the migrations against it under `POSTGRESQL_SCHEMA=mcp_test`. The tests assert that tables and indexes are created in `mcp_test` (and NOT in `public`), that functions remain in `mcp_test` for CVE-2018-1058 purposes, that the compression and chunking migrations are idempotent when applied twice under the non-default schema, that a mixed-case schema is quoted both in function DDL and in the runtime `jsonb_merge_patch()` call, and that pooled connections keep the configured `search_path` across the pool's `RESET ALL`. These tests run alongside the default-schema (`public`) tests, so both branches of the contract are exercised end to end on every PostgreSQL integration run.

## Migration CLI

For installations with existing fp32 embeddings, the `mcp-context-server-migrate` console script provides `--compress` and `--decompress` flags that re-encode the data in place. The CLI is single-backend (SQLite or PostgreSQL); cross-backend migration is handled by the existing v2-to-v3 migration path first, then `--compress` on the target.

The CLI is shipped alongside the server (entry point: `app.cli.migrate:main`) and is on `PATH` after a normal `uv` or `pip` install.

### Backup required

> [!WARNING]
> The `--compress` and `--decompress` operations rewrite the embedding storage and drop the previous table at the end of an atomic transaction. **Back up your database before running the CLI.** For SQLite, take a filesystem copy of the `.db` file. For PostgreSQL, run `pg_dump`.

Decompression is **lossy by design**: the decoded vectors approximate the original MSE component (for `variant='mse'`) and cannot perfectly reconstruct the inner-product information (for `variant='ip'`). Round-tripping `--compress` then `--decompress` does not recover the original fp32 vectors. If you need the original values, restore from backup.

### Usage

Always run `--dry-run` first to preview the plan:

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress \
  --dry-run
```

The dry run prints the BACKUP REQUIRED banner, the source URL (with credentials masked), the row count, the destination table name, the singleton provenance values, and an extrapolated execution time computed from a small probe batch. It performs no writes.

After verifying the dry-run output, re-run without `--dry-run` to execute the encode-and-replace:

```bash
# All four export lines are optional in v3.0.0; the values shown are the defaults.
# Override only if you intentionally want a different configuration.
# export ENABLE_EMBEDDING_COMPRESSION=true
# export COMPRESSION_BITS=4
# export COMPRESSION_VARIANT=ip
# export COMPRESSION_SEED=0  # immutable after first compressed row is written

mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress
```

The CLI streams fp32 rows in batches of 10 000 (the module-level `MIGRATION_BATCH_SIZE` constant in `app/cli/migrate_compression/storage.py`), encodes each batch through the compression provider, and INSERTs each batch into `vec_context_embeddings_compressed`. The singleton `compression_metadata` INSERT and the source `DROP TABLE` (and, on PostgreSQL, the `idx_vec_context_embeddings_hnsw` HNSW index DROP) run LAST inside a single `begin_transaction()` that wraps every read, encode, and write of the data movement itself. Peak working set is bounded by `O(batch_size * dim * 4)` bytes for the fp32 read window -- approximately 40 MB at the default `batch_size=10000` and `dim=1024` -- independent of the total row count, so memory consumption does not scale with corpus size. If any step before COMMIT fails, the transaction rolls back and the original `vec_context_embeddings` table remains intact; on PostgreSQL the `vec_context_embeddings_compressed` and `compression_metadata` table creation shares that same transaction (transactional DDL), so a rollback there truly leaves no trace. On SQLite it does not: creating those two tables is idempotent `IF NOT EXISTS` DDL that runs via `executescript()` before `BEGIN IMMEDIATE` opens, so it autocommits ahead of the streamed transaction. An aborted `--compress` on SQLite can therefore leave three pieces of schema behind even though no rows were ever written: the empty `vec_context_embeddings_compressed` and `compression_metadata` tables, and -- on a database upgraded in place from before the column existed -- the nullable `compression_metadata.codebook_fingerprint` column, added by the same pre-`BEGIN` step. None of it is harmful: the fp32 source table is untouched, the table creation is a no-op against the leftovers so a re-run proceeds normally and performs the full encode, and the startup compression migration applies the same idempotent `ALTER` anyway. Note that with `ENABLE_EMBEDDING_COMPRESSION=true` already exported (as the instructions above direct before running `--compress`), a server started on the aborted state refuses to boot with exit 78 and tells you to run `--compress` -- the fp32 rows are still uncompressed and the provenance row was never written. The batch size is a module-level constant rather than a CLI flag or environment variable; operators who need a different value re-run the CLI after editing the constant.

After the migration completes, start the server with the same env vars set. The server reads the seed from the database row and serves search through the compressed read path.

### Reversing the compression

`--decompress` is the symmetric reverse: it decodes every compressed row, recreates the fp32 `vec_context_embeddings` table, and (on PostgreSQL) recreates the `idx_vec_context_embeddings_hnsw` HNSW index. It follows the same streaming + single-transaction shape as `--compress`: compressed rows are read in batches of 10 000, decoded per batch via the variant's decode path, and INSERTed into the recreated fp32 table. The source compressed table DROP and the HNSW index recreation run LAST inside the same atomic transaction. The same peak-memory bound and the same all-or-nothing rollback contract apply to the data movement -- and the same SQLite qualification applies to the schema step: the `vec_context_embeddings` `CREATE VIRTUAL TABLE` autocommits via `executescript()` before `BEGIN IMMEDIATE` opens, so an aborted `--decompress` on SQLite can leave that table present and empty (harmless: it is IF NOT EXISTS DDL, and re-running `--decompress` proceeds normally against it). On PostgreSQL the equivalent DDL shares the transaction and rolls back with everything else.

With ZERO compressed rows, `--decompress` takes a zero-data path instead: it drops the empty compressed table and clears the provenance row WITHOUT provisioning any fp32 infrastructure. This matters on deployments whose embedding storage (`embedding_chunks`, sqlite-vec, the pgvector extension) was never provisioned -- for example a database that gained a compression schema while running with `ENABLE_EMBEDDING_GENERATION=false` under an earlier release -- where the full reverse path would fail on the missing infrastructure. The next startup provisions fp32 storage per `ENABLE_EMBEDDING_GENERATION` as usual -- and regardless of that toggle when the database carries embedding infrastructure (an `embedding_metadata` table), so the cleanup paths that gate on it are never stranded on a missing fp32 table.

Before running it, unset or disable `ENABLE_EMBEDDING_COMPRESSION` so the validator does not reject the post-decompression startup:

```bash
unset ENABLE_EMBEDDING_COMPRESSION

mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --decompress
```

Reconstruction is lossy; use this flag only when you intend to abandon compression on a given database.

### Idempotency

Running `--compress` against an already-compressed database is safe. The CLI detects the existing singleton `compression_metadata` row, prints `[INFO] compression_metadata row already present`, and exits with status 0 without modifying any data. The same idempotency applies to `--decompress` when there is no compression to undo.

### Expected impact

- Table changes: the fp32 `vec_context_embeddings` table is dropped; `vec_context_embeddings_compressed` and `compression_metadata` are populated. On PostgreSQL, the `idx_vec_context_embeddings_hnsw` HNSW index is dropped during `--compress` and recreated during `--decompress`.
- Peak memory: bounded by `batch_size * dim * 4` bytes (approximately 40 MB at the default `batch_size=10000` and `dim=1024`), independent of total row count. The migration streams rows in batches rather than reading the entire fp32 table into memory at once.
- Latency cost at default configuration: roughly 445-477 microseconds per encode on a single CPU thread (d=1024, bits=4, variant='ip'). Total run time scales linearly with row count.
- Recall impact: the project's recall regression gate requires at least 0.85 top-K overlap with fp32 ground truth across the supported `(bits, variant)` cells; measured recall is 1.0 in the test corpus.

For the wider v2-to-v3 migration context (including the integer-to-UUIDv7 schema migration that runs separately), see the [Migration Guide](migration-v2-to-v3.md).

### Backfilling missing embeddings

For entries that exist in `context_entries` but have no corresponding `embedding_metadata` row (typically left over from a cross-backend migration where vector embeddings were dropped, or from a corpus imported without embeddings), the `--embed-missing` flag regenerates embeddings via the configured embedding provider:

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --embed-missing \
  --dry-run
```

The dry run reports the number of entries that lack embeddings and hold no generation job, and the pending and failed generation jobs the command completes first (see [Completing deferred generation jobs](#completing-deferred-generation-jobs)), without calling a provider. Re-run without `--dry-run` to execute the backfill. The CLI streams entries one at a time inside per-entry transactions; a failure mid-backfill does not poison the overall run, and re-running the command is idempotent (entries that already have embeddings are skipped).

When `ENABLE_EMBEDDING_COMPRESSION=true` (the default in v3.0.0), the backfill writes compressed payloads to `vec_context_embeddings_compressed`. When `ENABLE_EMBEDDING_COMPRESSION=false`, it writes fp32 vectors to `vec_context_embeddings`.

`--embed-missing` fills ONLY the entries that lack embeddings and leaves existing embeddings untouched. To prevent silently mixing incompatible embedding spaces, it runs a pre-flight check: if the database already contains embeddings produced by a different `EMBEDDING_MODEL`, or recorded at a different `EMBEDDING_DIM`, than the configured values, the CLI refuses with exit code 1 and explains the fix. The canonical post-cross-backend-migration case (where ALL embeddings were dropped) has no existing embeddings, so the check passes and the backfill proceeds. To re-embed an entire corpus under a NEW model, use `--re-embed` (below) instead.

`--embed-missing` may be combined with `--compress` for a one-shot compress-then-backfill workflow against a database that has both fp32 embeddings AND entries lacking embeddings:

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress \
  --embed-missing
```

The compose order is fixed: `--compress` runs first (converts the existing fp32 data), then `--embed-missing` backfills any missing entries directly into the compressed layout. If `--compress` succeeds because the database is already compressed (idempotent no-op), the flow still falls through to `--embed-missing` correctly.

`--embed-missing` is NOT combinable with `--decompress`. If you need to decompress an existing database AND backfill missing entries, run the two commands separately (decompress first, then run `--embed-missing` with `ENABLE_EMBEDDING_COMPRESSION=false`).

Cost note: `--embed-missing` calls whichever embedding provider is configured via `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`. For local Ollama the cost is zero; for cloud providers (OpenAI / Voyage / HuggingFace) the cost scales with the number of missing entries. The CLI prints a warning banner before starting that lists the source URL (with credentials masked) and reminds the operator that live provider calls will follow.

### Completing deferred generation jobs

A store or update made with `defer_generation=true` leaves a generation job beside its entry until a server's generation worker generates the entry's embeddings, summary and index-tree node summaries. `--embed-missing` and `--re-embed` complete those jobs first, in their own process, so a database whose server is stopped still receives every deferred output. The command returns every failed job to the queue with its attempts reset, then claims the oldest claimable job under a lease, as a server's worker does, runs it with the configured embedding provider and, when `ENABLE_SUMMARY_GENERATION=true`, the configured summary provider, and repeats until no job is claimable. Its own pass then covers only the entries that hold no job.

- **Attempts.** The run ends as soon as no job is claimable and never waits for a backoff. A job whose attempt fails is claimable again once its backoff ends (see [Deferred Generation Settings](environment-variables.md#deferred-generation-settings)), so a run still working through other jobs by then attempts it again, until it completes or its last attempt (`GENERATION_JOB_MAX_ATTEMPTS`) fails it. A job a running server holds stays with that server. The entry of every job that remains keeps its job and is left out of the command's pass, so it is neither backfilled nor re-embedded.
- **Counts.** The dry run calls no provider and prints `Found <p> pending and <f> failed generation jobs.` beside the entry count, which counts only entries that hold no job. A run prints the outcome of each attempt it made, so a job attempted twice counts twice: how many attempts completed their job, left it for a later attempt, failed it and were discarded, and how many failed jobs it returned to the queue first. A discarded attempt stored nothing, because the job's entry was deleted, or its text or the claim changed, while it ran. When jobs remain, it also prints how many are pending and how many failed.
- **Exit status.** The command exits with status 1 before it touches any job when a provider it needs is unavailable: the embedding provider, and the summary provider when summary generation is enabled and jobs exist. A job that fails does not change the exit status: the run exits 0 and counts it, and its entry stays readable and FTS-searchable. An unexpected failure exits with status 2.
- **Older databases.** A database that no release with deferred generation has started yet has no `generation_jobs` table; the command then finds no jobs and covers every entry.

Run the command with the provider and model settings the server uses, because the jobs it completes are generated with them.

### Re-embedding the whole corpus (changing the embedding model)

`--embed-missing` does not touch entries that already have embeddings, so it cannot change the embedding MODEL of an existing corpus. To switch models -- for example, replacing `qwen3-embedding:0.6b` with a different model of the same dimension -- use `--re-embed`. It regenerates embeddings for EVERY `context_entries` row (deleting the old vectors first) using the currently configured `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`, and backfills any entries that were missing embeddings along the way (it is a superset of `--embed-missing`):

```bash
# Point EMBEDDING_MODEL at the new model, keep EMBEDDING_DIM unchanged, then:
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --re-embed \
  --dry-run
```

The dry run reports the number of entries that would be re-embedded, the pending and failed generation jobs the command completes first, and the existing model(s) being replaced, without calling a provider. An entry whose generation job remains after that step is not re-embedded. Re-run without `--dry-run` to execute. Each entry's delete + regenerate runs inside one transaction, so an entry is never left without embeddings. `--re-embed` works on both fp32 and compressed layouts and requires `ENABLE_EMBEDDING_GENERATION=true`. It is mutually exclusive with `--compress` / `--decompress`, and it supersedes `--embed-missing` (re-embedding the whole corpus already covers the gaps).

`--re-embed` deliberately refuses a DIMENSION change: if the configured `EMBEDDING_DIM` differs from the stored dimension, it exits with an actionable error. A dimension change rewrites the vector-storage geometry -- the fp32 vector column width is fixed at table creation, and under compression the dimension is part of the seed-locked `compression_metadata` codebook -- so it requires the destructive rebuild documented in the [Migration Guide](migration-v2-to-v3.md#changing-the-embedding-model-or-dimensions), not an in-place re-embed.

## Bonus Benefits

### pgvector dimension limit workaround

The native `pgvector` `vector(D)` HNSW index rejects dimensions above 2000, which blocks models like OpenAI's `text-embedding-3-large` (3072 dimensions) from using HNSW search on plain pgvector. Embedding compression sidesteps this limit because the compressed table uses BYTEA (not `vector(D)`); the read path uses the TurboQuant inner-product estimator directly, not pgvector operators. PostgreSQL deployments running high-dimensional models can use the compressed read path without HNSW and without hitting the 2000-dimension ceiling.

## Byte-Alignment Matrix

The bit-packing layout requires `dim * effective_bits` to be a multiple of 8, where `effective_bits` is **variant-dependent**: for `variant='ip'`, `effective_bits = COMPRESSION_BITS - 1` (one bit per coordinate is reserved for the QJL sign); for `variant='mse'`, `effective_bits = COMPRESSION_BITS`. This means the alignment constraint differs between the two variants when `COMPRESSION_BITS - 1` and `COMPRESSION_BITS` straddle a factor-of-8 boundary for a given `dim`. A violated constraint is detected by the startup validator before any data is written and raises `ConfigurationError` (exit 78); the server does not start. All common embedding dimensions (those that are multiples of 8) satisfy the constraint for every supported `(bits, variant)` combination:

| dim  | bits=2 (ip: eff=1, mse: eff=2) | bits=3 (ip: eff=2, mse: eff=3) | bits=4 (ip: eff=3, mse: eff=4) |
|------|--------------------------------|--------------------------------|--------------------------------|
| 768  | OK (768, 1536)                 | OK (1536, 2304)                | OK (2304, 3072)                |
| 1024 | OK (1024, 2048)                | OK (2048, 3072)                | OK (3072, 4096)                |
| 1536 | OK (1536, 3072)                | OK (3072, 4608)                | OK (4608, 6144)                |
| 2048 | OK (2048, 4096)                | OK (4096, 6144)                | OK (6144, 8192)                |
| 3072 | OK (3072, 6144)                | OK (6144, 9216)                | OK (9216, 12288)               |

Each cell shows `OK (ip_product, mse_product)` where the listed products are `dim * effective_bits` for the `ip` and `mse` variants respectively; all are multiples of 8. Non-standard dimensions whose `dim * effective_bits` is not divisible by 8 for the configured variant are rejected at startup (exit 78) before any data is written. If you use a non-listed dimension, verify that `dim * (COMPRESSION_BITS - 1)` is a multiple of 8 for `variant='ip'` or that `dim * COMPRESSION_BITS` is a multiple of 8 for `variant='mse'`; if not, choose a different `dim` or `bits` value that satisfies the constraint for the active variant.

## Troubleshooting

### ConfigurationError exit 78 at startup

The startup validator detected a mismatch between the runtime `CompressionSettings` and the persisted `compression_metadata` row. The validator compares each of the five config fields recorded at first bootstrap -- `provider`, `bits`, `variant`, `seed`, and `dim` -- and the error message lists every field that disagrees. When those five match, it additionally re-derives the realized `codebook_fingerprint` (the `numpy.linalg.qr` rotation digest) and raises here too if it diverges -- the signature of a cross-host BLAS/LAPACK/CPU QR difference that would silently corrupt decode/search. The supervisor will not auto-restart on exit 78.

Resolution:

- Identify the canonical configuration. The values in `compression_metadata` are the database's record of what the data was encoded with; changing those values would corrupt every payload.
- Align the runtime env vars to match the persisted row.
- In multi-pod deployments, ensure every pod inherits the same `COMPRESSION_SEED` (use a ConfigMap-bound env var).

### Server starts with `compression_metadata` empty

If `ENABLE_EMBEDDING_COMPRESSION=true` is set but the database does not yet have a `compression_metadata` row, the startup migration creates the table and the validator inserts the singleton row from the active env vars. This is the normal first-start path. If the bootstrap fails, check the server log for the underlying migration error; common causes are insufficient DDL privileges on PostgreSQL.

### Slow queries on compressed data

The compressed read path is a memory-bounded linear scan. If queries are slow, check that the candidate filters (`thread_id`, `tags`, `metadata`, date range) are narrowing the row set effectively. Linear-scan latency grows linearly with the number of candidate chunks; sub-second performance is comfortable up to roughly 100 000 to 1 000 000 chunks per thread.

For very large corpora that exceed the linear-scan ceiling, approximate nearest-neighbor indexing over compressed payloads is planned for a future release (see [Future Capabilities](#future-capabilities)).

### `--compress` exits with `vec_context_embeddings not present`

The source database does not have an fp32 embedding table to compress. This typically means the database was provisioned with compression already enabled, or `--compress` was already applied and the legacy table was dropped. Verify the current state by inspecting `compression_metadata`; if a singleton row is present, the database is already compressed and no further action is needed.

## Future Capabilities

The following capabilities are planned for future versions and are not part of v3.0.0:

- Approximate nearest-neighbor (ANN) search over compressed payloads via a `pgvector` bit-Hamming HNSW prefilter plus the TurboQuant inner-product reranker. This will lift the linear-scan scale ceiling for very large corpora.
- Additional compression providers and additional bit widths (1-bit binary; bits 5 through 8). The current storage abstraction permits new providers without further schema changes.
- A dedicated `turbovec` storage backend (`STORAGE_BACKEND=turbovec`) as a sidecar option for 10M-plus chunk deployments.
- Per-chunk seed alternatives for advanced multi-pod patterns that need finer-grained independence than the current single-seed invariant allows.

## Related Reading

- Original TurboQuant paper: <https://arxiv.org/html/2504.19874v1> (Algorithm 1 and Algorithm 2; Theorem 1 distortion bound; Theorem 2 unbiased IP).
- [Environment Variables Reference](environment-variables.md) (Embedding Compression Settings section).
- [Migration Guide (v2 to v3 schema and compression)](migration-v2-to-v3.md).

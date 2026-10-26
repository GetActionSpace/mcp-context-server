# Migrating to the UUIDv7 Schema

This guide applies to the v2 -> v3 transition. Future major-version migrations will be documented in separate guides.

The context-entry primary key is a UUIDv7 value. On SQLite the `context_entries.id` column is `TEXT NOT NULL UNIQUE`; on PostgreSQL it is a native `UUID NOT NULL PRIMARY KEY`. Public IDs are exchanged at MCP tool boundaries as 32-character lowercase hex strings (no hyphens).

If you previously ran an installation whose `context_entries.id` column was an INTEGER (SQLite) or `BIGSERIAL` (PostgreSQL) and you want to keep your existing data, you must run the `mcp-context-server-migrate` CLI on a backup of your source database before pointing the new server at it. This file walks through that process.

## When You Need to Migrate

You need to migrate if all of the following apply:

- You ran a previous installation of this server and have a SQLite database file or a PostgreSQL instance with stored context entries.
- The `context_entries.id` column in that database is INTEGER (SQLite) or BIGINT / BIGSERIAL (PostgreSQL).
- You want to keep your existing data when upgrading to the UUIDv7-keyed schema.

If you are starting fresh (no existing data) you do NOT need to migrate. Install the new server, point it at an empty database location, and proceed normally.

## Pre-Migration Checklist

Before running the CLI:

- **Stop the server.** The migration tool requires exclusive read access to the source database.
- **Make a backup copy of the source database.** The migration reads the source in read-only mode (SQLite uses the `mode=ro` URI parameter; PostgreSQL uses `BEGIN TRANSACTION READ ONLY`), but a backup is still essential in case of an interrupted run or operator error.
- **Provision an empty target database.** For SQLite, a non-existent file path is fine -- the CLI creates the file and applies the current schema. It must be EMPTY or non-existent, not a database a v3 server has already initialized: compression is default-ON in v3, so such a target already carries compression provenance, and the CLI REFUSES the run rather than copy fp32 vectors the next server start would destroy (a later `--compress` cannot rescue them -- it is a no-op while a provenance row exists). If you must reuse an initialized target, run `mcp-context-server-migrate --source-url <target> --decompress` against it first to clear its compression provenance, then rerun the migration and finish with `--compress`. For PostgreSQL, create an empty database manually (for example, `CREATE DATABASE mcp_context_v3;`); the CLI does not create the PostgreSQL database itself, but it now **auto-initializes the target schema** (tables, indexes, and functions) on an empty target, mirroring the SQLite path. You no longer need to start the server once against the target to create the schema -- if the target has no `context_entries` table, the CLI builds it before copying. When the source carries embeddings, the auto-initialized target gets the **fp32** vector layout (the CLI never enables compression during initialization); enable compression afterward with the separate `--compress` step (see [Compressing an Existing Database](#compressing-an-existing-database)).
- **For SQLite source databases that use semantic search or FTS**, ensure the `sqlite-vec` extension is available in the Python environment running the CLI. The CLI loads the extension on both source and target connections when vector tables are present.
- **For PostgreSQL targets that should use semantic search after migration**, ensure the `pgvector` extension is enabled. When the source has embeddings, the CLI runs `CREATE EXTENSION IF NOT EXISTS vector` on the target during auto-initialization; this requires sufficient privileges. On managed services that restrict `CREATE EXTENSION` (notably Supabase), enable pgvector first via the database management interface (Supabase: Dashboard -> Database -> Extensions -> vector), then rerun.
- **Protect PostgreSQL credentials.** Avoid placing passwords directly on the command line. Prefer environment-variable substitution or a `.pgpass` file. URLs printed by the CLI are masked in stdout (`postgresql://user:***@host/db`), but passwords on the original command line remain visible in process listings and shell history.

## CLI Usage

### Installation

The CLI is installed alongside the server. If you installed `mcp-context-server` via `uv` or `pip`, `mcp-context-server-migrate` is on your PATH (it is registered in `[project.scripts]`). Verify with:

```bash
mcp-context-server-migrate --help
```

### Same-Backend Migration (SQLite to SQLite)

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/source.db \
  --target-url sqlite:////path/to/target.db
```

The bare filesystem path form is also accepted:

```bash
mcp-context-server-migrate \
  --source-url /path/to/source.db \
  --target-url /path/to/target.db
```

Note that `sqlite:///` requires three slashes followed by an absolute path. The CLI also accepts plain filesystem paths without a scheme.

### Same-Backend Migration (PostgreSQL to PostgreSQL)

```bash
mcp-context-server-migrate \
  --source-url postgresql://user:password@host:5432/old_db \
  --target-url postgresql://user:password@host:5432/new_db
```

Passwords in printed URLs are masked in stdout output (`postgresql://user:***@host/db`).

### PostgreSQL Connection Modes (Poolers, Schema, SSL)

The CLI honors the same `POSTGRESQL_*` environment variables the server uses for every PostgreSQL connection it opens (source, target, and the auto-initialization connection):

- **Schema (`POSTGRESQL_SCHEMA`).** The CLI applies `search_path = "<POSTGRESQL_SCHEMA>", public` on every connection, so a non-default schema is resolved correctly for both reads and the auto-initialized target. Set `POSTGRESQL_SCHEMA` to the same value the server will use.
- **Connection poolers (`POSTGRESQL_STATEMENT_CACHE_SIZE`).** The CLI sends no PostgreSQL startup parameters (it applies `search_path` and the other session parameters with `SET` after connecting), so no pooler allowlist entry is needed for it. It does use prepared statements by default (cache size 100). Transaction-mode poolers -- PgBouncer transaction mode, Pgpool-II, AWS RDS Proxy, and the **Supabase Transaction Pooler (port 6543)** -- break prepared statements. For those, either run the migration through a session-capable endpoint (a Direct connection or the Supabase **Session Pooler**, both on port 5432) or set `POSTGRESQL_STATEMENT_CACHE_SIZE=0`. For Supabase specifically, prefer the Direct connection or Session Pooler URL on port 5432.
- **SSL.** SSL is taken from the connection URL. Put `?sslmode=require` (or your chosen mode) directly in `--source-url` / `--target-url`; the CLI does not inject `sslmode` from settings because the source and target may be different databases than the running server.

### Cross-Backend Migration

The CLI supports SQLite -> PostgreSQL and PostgreSQL -> SQLite. Vector embeddings (the binary `embedding` BLOBs in SQLite's `vec_context_embeddings` table and the corresponding PostgreSQL `pgvector` values) are NOT portable across backends because their binary representations differ. When cross-backend migration is requested, the CLI emits a warning and DROPS **only** the vector embeddings -- you must re-embed the target database after migration with `mcp-context-server-migrate --embed-missing` (see [Cross-Backend Vector Embedding Warning](#cross-backend-vector-embedding-warning)).

All other data is copied: `context_entries` (with `summary` and `content_hash`), `tags`, and `image_attachments` (image payloads are portable -- BYTEA on PostgreSQL maps to BLOB on SQLite). The target schema is auto-initialized when absent (the SQLite target via the existing initializer; the PostgreSQL target via the base schema, without the vector tables, which the server creates at your configured `EMBEDDING_DIM` when you re-embed). Full-text search is provisioned on the target regardless of direction even though the FTS data itself is not portable: for PostgreSQL -> SQLite the SQLite target's FTS5 index is rebuilt from the copied rows, and for SQLite -> PostgreSQL the PostgreSQL target's `text_search_vector` is a generated column created during target initialization, so it is populated automatically as rows are copied (no separate rebuild step).

On every migration path -- same-backend and cross-backend alike -- each copied `context_entries` row is stamped with the access-control columns the source predates: `owner_id` is set to the configured `ACCESS_CONTROL_DEFAULT_PRINCIPAL` (default `local`) and `visibility` to `private` (fail-closed). If your deployment uses a non-default principal, set `ACCESS_CONTROL_DEFAULT_PRINCIPAL` in the environment BEFORE running the CLI so migrated rows are owned by the right identity. See [Access Control Settings](environment-variables.md#access-control-settings).

#### Rows a PostgreSQL Target Cannot Accept

A SQLite source may hold a value that SQLite stores happily but a PostgreSQL target cannot bind. Two byte sequences are unstorable in any column: an embedded NUL (`U+0000`) and an unpaired UTF-16 surrogate (a code point that cannot be encoded as UTF-8). The `jsonb` columns -- `context_entries.metadata` and `image_attachments.image_metadata` -- carry two further hazards: a metadata NUL serialized as its six-character JSON escape sequence (which the target's `::jsonb` cast rejects even though the raw bytes contain no literal NUL), and metadata that is not valid JSON at all (which the `::jsonb` cast also rejects). Any of these, bound blindly, would abort the whole cross-backend transaction mid-run and roll back every already-copied row.

A THIRD class is about the target's INDEXES rather than its column types. PostgreSQL refuses to store an index tuple larger than roughly 2.7 KB, and it evaluates a metadata expression index -- including the SQL cast an `integer`/`boolean` `METADATA_INDEXED_FIELDS` type hint puts in it -- on every INSERT. SQLite's equivalent indexes have no size limit and apply no cast, so a source predating the write-path caps can hold an oversized `thread_id`, an oversized tag, or an indexed metadata value the target cannot index at all (an oversized one, or `{"priority": "high"}` under an `integer`-typed field). The same pre-check also skips a value that is not a valid number under a `float`-typed field: the `float` index reads JSON numbers only and could hold it, but the field's declared type is `float`, and the server refuses such a value on both backends. This class is checked in EVERY direction that writes to PostgreSQL, including PostgreSQL to PostgreSQL: the metadata expression indexes are deliberately absent from the base schema, so a source that was only ever initialized by this CLI can carry such a value even though it is itself a PostgreSQL database.

Rather than crash, the CLI pre-checks each source row and SKIPS an offending row instead of aborting: it records the source row id and the offending column in the run's `errors`, and skips that row's `tags` and `image_attachments` children too (with a warning) so no child is left referencing a never-inserted parent (which would itself raise a foreign-key violation on PostgreSQL). Clean rows in the same run still migrate and the transaction commits. `--dry-run` surfaces every such row without inserting, so you can preview the full set before a real run.

Because each skipped row is recorded as an error, a run that skips any row exits with a nonzero status (exit `1`), so a scripted invocation does not mistake a partial migration for a clean one. Inspect the printed summary (or the `--report PATH` JSON) for the `column ... skipped: ...` entries, repair the offending values at the source -- shorten an oversized one, correct a value its field's type hint rejects -- and re-run to migrate the previously skipped rows.

### Optional Flags

- `--dry-run` -- run the full migration logic in memory (read source, build the integer-to-UUIDv7 mapping, compute target rows) but issue zero INSERT statements against the target. Useful for validating that the source database is well-formed before committing to a real run.
- `--report PATH` -- after the run completes, write the run's statistics (rows migrated, references rewritten, orphan references detected, warnings, errors) as JSON to `PATH`. If omitted, a human-readable summary is printed to stdout.

## What the CLI Does

The CLI performs the following steps in order:

1. **Read source rows in deterministic order.** Reads from `context_entries` ordered by `(created_at ASC, id ASC)`. The deterministic ordering ensures the in-memory integer-to-UUIDv7 mapping is reproducible across runs.

2. **Generate a UUIDv7 per row from each row's `created_at`.** Uses `uuid_utils.uuid7(timestamp=int(created_at.timestamp()), nanos=created_at.microsecond * 1000)`. The `timestamp` parameter is in UNIX SECONDS with an optional nanosecond fraction; passing milliseconds in the `timestamp` slot would shift the embedded timestamp roughly 1000x further into the future, producing UUIDs whose decoded year is around 50,000 AD. Upstream tracker documenting this confusion source: [`aminalaee/uuid-utils#73`](https://github.com/aminalaee/uuid-utils/issues/73). The resulting UUIDs preserve the source's chronological order: lex-string comparison on the 32-character lowercase hex form is monotonic with respect to `created_at` at millisecond granularity. Sub-millisecond ordering is determined by the UUID random tail, which is acceptable for the server's deduplication interleaving check.

3. **Rewrite `metadata.references.context_ids` arrays.** For each row, walks the parsed `metadata` JSON recursively, finding every `references.context_ids` array. Integer entries are replaced with the mapped UUIDv7 hex string. String entries are preserved unchanged (treated as already-UUID values). Integer entries that do not match any source row are kept as integers and flagged in the run summary as `orphan_references` (see Troubleshooting below). Malformed shapes (for example, a non-array `context_ids` value) are flagged as `malformed_references` errors, and the row's metadata is copied verbatim.

4. **Copy free-form text byte-for-byte.** The `text_content` and `summary` columns are copied unchanged. Any mention of an integer ID inside the prose (for example, "see entry 8944") is left as-is. The CLI does not attempt to rewrite free-form text -- such substrings become inert references after migration, which is the intended outcome (eliminating false positives from regex-style ID detection across the corpus).

5. **Copy tags, image attachments, embedding metadata, embedding chunks, and vector rows.** Only the `context_id` foreign-key column is remapped to the new UUIDv7 hex value. Internal embedding identifiers (`embedding_chunks.id`, `embedding_chunks.vec_rowid`, vec0 rowids, `vec_context_embeddings.id`) are preserved verbatim because the project's chunking-layer architecture uses INTEGER bridges to vec0; only the outer `context_id` foreign key changes type.

6. **Rebuild the SQLite FTS5 index.** When the source has `context_entries_fts`, the CLI issues `INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')` against the target to re-index the copied rows. The FTS5 rowid alignment with `context_entries.rowid_int` is preserved.

7. **Print or write the run report.** A summary of counters (rows migrated, references rewritten, orphan references, malformed references, tags/images/embeddings copied, FTS rebuild status, warnings, errors) is printed to stdout. If `--report PATH` is set, the same data is also written as JSON.

## Optional: PostgreSQL 18+ Server-Side UUIDv7 Generation

If you are running PostgreSQL 18 or later, you have the OPTION to use the server-side `uuidv7()` built-in for the `id` column default:

```sql
ALTER TABLE context_entries ALTER COLUMN id SET DEFAULT uuidv7();
```

This is a pure operator-side optimization and is NOT required for correct operation. The application generates UUIDs Python-side via the `app.ids` module by default; the column default would only matter if external clients ever insert rows without specifying an `id`. Most installations should leave this column default unset. PostgreSQL 17 and earlier do not have the `uuidv7()` function; users on those versions cannot apply this optimization.

## Troubleshooting

### Orphan References

If the run summary shows `orphan_references > 0`, your source database contained `metadata.references.context_ids` entries pointing to IDs that do not exist in `context_entries` (most commonly from older rows that were deleted before migration). The orphan IDs are preserved as integers in the migrated metadata; downstream consumers that interpret `references.context_ids` should be prepared to encounter both string UUIDs and stray integers in legacy data. To find and clean orphan references, query the migrated database after the run -- the JSON `references.context_ids` arrays are still queryable with `array_contains` operators.

### Source Database File Missing

If the CLI prints `source database file does not exist: <path>`, the path passed to `--source-url` does not exist on disk. For SQLite, ensure the path is correct and absolute. For `sqlite:///` URLs, note that three slashes are required: `sqlite:///C:/Users/me/db.sqlite` on Windows or `sqlite:////home/me/db.sqlite` on Linux.

### Target Database Already Has Data

The CLI refuses to write to a non-empty SQLite target. If you see `target database already contains context_entries rows: <path>`, delete the target file (or specify a different `--target-url`) and rerun.

### Recovering From an Interrupted Migration

The CLI does not currently support resume-from-checkpoint. If a migration is interrupted mid-run (Ctrl+C, process kill, machine power loss, network drop on PostgreSQL), the source database is unaffected -- it was opened read-only -- and the target database is left in one of two recoverable states described below.

The recovery procedure is the same in both cases: delete the target database and rerun the same `mcp-context-server-migrate` command. The source is unchanged, so a fresh run reads the same input data and produces the same target. Note that the lower 74 random bits of each UUIDv7 will differ from any prior interrupted run, but the embedded millisecond timestamps and the row-by-row chronological ordering are identical, which is the only property the server depends on.

**State 1: Empty or schema-only target.**

The data-copy transaction rolled back. The target file (SQLite) or database (PostgreSQL) may contain the empty schema (tables defined but no rows) or may not exist at all, depending on how far the CLI got before the interrupt. Recovery:

- SQLite: `rm /path/to/target.db` -- then rerun. Alternatively, pass a different `--target-url` to the rerun.
- PostgreSQL: `DROP DATABASE new_db; CREATE DATABASE new_db;` -- then rerun. Alternatively, point `--target-url` at a different empty database.

**State 2: Data committed but FTS index stale (SQLite only).**

There is a narrow window after the main data transaction commits but before the FTS5 rebuild commits. If the interrupt happens inside that window, the target contains all migrated rows but its `context_entries_fts` virtual table is stale or empty. This is intentional: the FTS rebuild runs outside the main data transaction so that a failing FTS rebuild does not destroy the row data. Two recovery options:

- Clean rerun: `rm /path/to/target.db` and rerun the CLI. This is the simplest option and is always correct.
- In-place FTS rebuild: run the SQLite shell command below to re-index the existing target without redoing the row copy:

  ```bash
  sqlite3 /path/to/target.db "INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')"
  ```

  This is the same idempotent rebuild directive that the CLI itself issues, so running it manually is safe.

PostgreSQL does not have a State 2 equivalent. Its `text_search_vector` column is a generated column maintained by the row INSERT, so a successful main transaction also produces a complete tsvector index.

**Preflight tip.** If you want to verify a migration will succeed before committing to a real run, pass `--dry-run` on a first invocation. The dry run reads the source, builds the integer-to-UUIDv7 mapping, and computes target rows in memory; it does NOT INSERT into the target and writes nothing to the target path. On SQLite the dry run operates against an ephemeral in-memory database, so it leaves no file (and no schema) at the target path; on PostgreSQL the target database is left empty. A subsequent real run against the same target proceeds normally -- no delete or rename is needed.

### Cross-Backend Vector Embedding Warning

If you ran a cross-backend migration (SQLite to PostgreSQL or vice versa) and saw the warning `cross-backend migration drops vector embeddings; re-embed the target after migration`, your target database contains all rows and metadata but no vector embeddings. The server never backfills embeddings on its own, so restore them with the migration CLI: run `mcp-context-server-migrate --source-url <target-url> --embed-missing` against the target database. The `--embed-missing` flag invokes the configured embedding provider (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`) for every entry that lacks an `embedding_metadata` row. When `ENABLE_EMBEDDING_COMPRESSION=true` is set (the default in v3.0.0), the new embeddings land directly in `vec_context_embeddings_compressed`; otherwise they land in `vec_context_embeddings`. Run with `--dry-run` first to count missing entries without calling the provider. See the [Backfilling missing embeddings](embedding-compression.md#backfilling-missing-embeddings) section of the Embedding Compression Guide for details.

### Second-Precision Source Data

If the run summary or log shows that the source database has second-precision `created_at` (no microsecond data), the CLI prints an informational log line. Same-second entries in the source are still uniquely identified (the UUIDv7 random tail differentiates them), but the resulting ordering of same-second entries may differ from the source's `id`-based ordering. This is acceptable per RFC 9562 and the server's deduplication invariants.

### `writable_schema` In-Place ALTER (do NOT use)

Some online guides suggest using SQLite's `writable_schema` pragma to ALTER the `context_entries.id` column type in-place from INTEGER to TEXT. **Do not do this.** It is documented as unsafe by the SQLite project, and it would not migrate the actual integer values to UUIDs anyway -- it would only change the column declaration while leaving every row's `id` as an integer, breaking referential integrity. The export-then-import path implemented by `mcp-context-server-migrate` is the correct approach.

## Post-Migration Verification

Suggested verification steps after a successful migration:

1. Start the server pointed at the target database. Confirm it starts cleanly (no schema-mismatch errors at startup).
2. Run a smoke test query: `list_threads` or a simple `search_context` call. Confirm the response includes 32-character hex `context_id` values.
3. If FTS was enabled on the source, run a small `fts_search_context` query and confirm results are returned.
4. If semantic search was enabled on the source AND the migration was same-backend (no embeddings dropped), confirm a `semantic_search_context` query returns expected results.
5. Spot-check that `metadata.references.context_ids` arrays contain UUIDv7 hex strings (not integers) in a few migrated rows.

A one-liner SQLite sanity check that confirms the `id` column is TEXT in the target:

```bash
python -c "import sqlite3; c = sqlite3.connect('target.db'); rows = c.execute(\"SELECT typeof(id) FROM context_entries LIMIT 1\").fetchall(); assert rows and rows[0][0] == 'text', rows"
```

## Compressing an Existing Database

If you want to enable the optional `ENABLE_EMBEDDING_COMPRESSION` feature on a database that already contains fp32 embeddings, run the `mcp-context-server-migrate` CLI with the `--compress` flag. This step is independent of the v2-to-v3 schema migration described above: if your database came from a v2 installation, run the v2-to-v3 migration first, then optionally run `--compress` on the resulting v3 database.

Order matters, and the CLI enforces it: `--compress` is a no-op once a `compression_metadata` provenance row exists, so it cannot rescue vectors copied into an already-compressed target. That is why a migration whose source carries embeddings is refused when the target is already configured for compressed embeddings (see the Pre-Migration Checklist). Migrate into an empty target first, then compress.

For background on what compression does, how it is configured, and the multi-pod seed-locked invariant, see the [Embedding Compression Guide](embedding-compression.md).

### Backup Required

> [!WARNING]
> The `--compress` flag is destructive: it permanently drops the `vec_context_embeddings` fp32 table and replaces it with `vec_context_embeddings_compressed`. The original fp32 vectors are NOT recoverable from the compressed payload (the compression is lossy by design). **Back up your database before running the CLI.**
>
> - SQLite: take a filesystem copy of the `.db` file.
> - PostgreSQL: run `pg_dump` to a portable archive.

### Dry Run First

Always run `--dry-run` before the real execution to preview the operation:

```bash
mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --compress \
  --dry-run
```

The dry run prints the BACKUP REQUIRED warning, the source URL (with credentials masked), the row count, the destination table name, the singleton provenance values that will be recorded, and an estimated execution time computed from a probe batch. No writes are issued.

### Execute the Compression

After verifying the dry-run plan, export the compression env vars and re-run without `--dry-run`:

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

The CLI streams fp32 rows in batches of 10 000, encodes each batch through the compression provider, and INSERTs each batch into `vec_context_embeddings_compressed`. The singleton `compression_metadata` INSERT and the source table DROP run LAST, all inside the same single atomic transaction. Peak working set is bounded by `O(batch_size * dim * 4)` bytes for the fp32 read window (approximately 40 MB at the default `batch_size=10000` and `dim=1024`), independent of total row count. On PostgreSQL the `idx_vec_context_embeddings_hnsw` HNSW index is dropped before the source table. If any step fails before COMMIT, the transaction rolls back end-to-end: no resume marker, and the source `vec_context_embeddings` table remains intact. On PostgreSQL that is literally all of it -- the compressed-table and provenance-table creation shares the transaction (transactional DDL) and leaves no trace. On SQLite that DDL is idempotent `IF NOT EXISTS` schema run through `executescript()` BEFORE `BEGIN IMMEDIATE` opens, so it autocommits ahead of the streamed transaction: an aborted run can leave the empty `vec_context_embeddings_compressed` and `compression_metadata` tables behind, plus the nullable `compression_metadata.codebook_fingerprint` column on a database upgraded in place from before that column existed. None of it is harmful -- no rows were written, the fp32 source is untouched, and a re-run performs the full encode against the leftovers.

Top-K recall against fp32 ground truth is at least `0.85` per query at the default configuration (the project's regression gate; measured recall is `1.0` on the test corpus).

### After Compression

Start the server with the same compression env vars exported. The server reads the seed from the `compression_metadata` row and serves search through the compressed read path. A mismatch between the env vars and the stored row raises `ConfigurationError` (exit 78) and refuses startup.

### Reversing the Compression

The `--decompress` flag reverses the operation, decoding the compressed payload back to fp32 and recreating the fp32 vec table. Reconstruction is **lossy**: the decoded vectors approximate the original MSE component (for `variant='mse'`) and cannot perfectly reconstruct the inner-product information (for `variant='ip'`). Use it only when you intend to abandon compression on a given database.

`--decompress` follows the same streaming + single-transaction shape as `--compress`: compressed rows are read in batches of 10 000, decoded per batch, and used to rebuild the fp32 layout directly from each compressed row's own `context_id`/`start_index`/`end_index`. On SQLite this rebuilds BOTH the recreated `vec_context_embeddings` table and the `embedding_chunks` bridge (linked through a fresh `vec_rowid`), so the reversal does not depend on any pre-existing `embedding_chunks` rows; on PostgreSQL the fp32 read path joins by `context_id`, so only `vec_context_embeddings` is recreated. The source compressed table DROP and (on PostgreSQL) HNSW index recreation run LAST inside the same atomic transaction. The same peak-memory bound and the same rollback contract apply, including the SQLite qualification: the `vec_context_embeddings` `CREATE VIRTUAL TABLE` likewise autocommits before `BEGIN IMMEDIATE`, so an aborted `--decompress` on SQLite can leave that table present and empty (harmless, and a re-run proceeds normally against it), while on PostgreSQL the equivalent DDL rolls back with everything else.

Unset `ENABLE_EMBEDDING_COMPRESSION` first so the post-decompression startup validator does not reject the disabled state:

```bash
unset ENABLE_EMBEDDING_COMPRESSION

mcp-context-server-migrate \
  --source-url sqlite:////path/to/db.sqlite \
  --decompress
```

On PostgreSQL the `idx_vec_context_embeddings_hnsw` HNSW index is recreated after the fp32 table is rebuilt.

### Cross-Backend Note

The compression CLI is single-backend. If you are migrating from SQLite to PostgreSQL (or vice versa), run the v2-to-v3 cross-backend migration first, then run `--compress` on the target.

### Idempotency

Running `--compress` against an already-compressed database is a no-op: the CLI detects the singleton `compression_metadata` row, prints an informational message, and exits successfully. The same applies to `--decompress` when there is no compression to undo.

### Further Reading

- Feature reference: [Embedding Compression Guide](embedding-compression.md)
- Configuration: [Environment Variables Reference](environment-variables.md#embedding-compression-settings)

## Changing the Embedding Model or Dimensions

There are two distinct scenarios, and they have different procedures because changing the model and changing the dimension affect the vector storage differently.

### Changing the model at the same dimension (use `--re-embed`)

Switching to a different embedding model that produces vectors of the SAME dimension (for example, replacing `qwen3-embedding:0.6b` with another 1024-dimensional model) is a one-command operation. `--re-embed` regenerates embeddings for EVERY `context_entries` row using the currently configured `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`, deleting the old vectors first and backfilling any entries that were missing embeddings along the way (it is a superset of `--embed-missing`). It works on both the fp32 and compressed layouts.

Steps:

1. **Stop the server** and **back up the database** (`--re-embed` deletes and rewrites every embedding).
2. **Set the new `EMBEDDING_MODEL`** in the environment, leaving `EMBEDDING_DIM` unchanged, and ensure `ENABLE_EMBEDDING_GENERATION=true`.
3. **Preview** with `--dry-run` -- it reports how many entries will be re-embedded, the pending and failed generation jobs of deferred writes the command completes first, and the existing model(s) being replaced, without calling a provider:

   ```bash
   mcp-context-server-migrate --source-url sqlite:////path/to/db.sqlite --re-embed --dry-run
   ```

4. **Execute** by re-running without `--dry-run`. The outstanding generation jobs are completed first (see [Completing deferred generation jobs](embedding-compression.md#completing-deferred-generation-jobs)); an entry whose job remains is not re-embedded. Each entry's delete + regenerate runs inside one transaction, so an entry is never left without embeddings. On a compressed database the new payloads land in `vec_context_embeddings_compressed`; on an fp32 database they land in `vec_context_embeddings`.
5. **Restart the server** with the new `EMBEDDING_MODEL` (and the same `COMPRESSION_*` values, if compression is enabled).

`--re-embed` requires `ENABLE_EMBEDDING_GENERATION=true`, is mutually exclusive with `--compress` / `--decompress`, and supersedes `--embed-missing`. Related: the `--embed-missing` flag now refuses to backfill into a database whose existing embeddings use a different model or dimension than the configured values (mixing embedding spaces would corrupt semantic search); it points you here when it detects a model change.

### Changing the dimension (destructive rebuild)

Changing `EMBEDDING_DIM` is heavier and `--re-embed` deliberately refuses it. The dimension is baked into the vector-storage geometry: the fp32 `vec_context_embeddings` column width is fixed at table creation, and under compression the dimension is part of the seed-locked `compression_metadata` codebook (immutable by design). A dimension change therefore requires recreating the vector storage from scratch, not an in-place re-embed. Follow the [Changing Embedding Dimensions](semantic-search.md#changing-embedding-dimensions) procedure: back up, update the configuration, delete the database (or drop and recreate the vector tables at the new dimension), restart the server to create fresh tables, and re-store the data so embeddings are generated at the new dimension.

## Reassigning Entry Ownership

`--reassign-owner FROM TO` is an in-place mode like `--compress`, `--decompress`, `--re-embed` and `--embed-missing`: it operates on the database named by `--source-url` and ignores `--target-url`. It rewrites the owner of every `context_entries` row owned by `FROM` to `TO` in one statement. Its main use is switching an existing deployment to `MCP_AUTH_PROVIDER=jwt`: the entries written over STDIO, `none` or `simple_token` are owned by `ACCESS_CONTROL_DEFAULT_PRINCIPAL` (default `local`), and under `jwt` no caller owns them: the private ones are unreachable and the public ones cannot be changed or deleted until they belong to an IdP subject (see [Switching an Existing Deployment to JWT](authentication.md#switching-an-existing-deployment-to-jwt)).

It is also the route for subjects the environment variable cannot hold. `ACCESS_CONTROL_DEFAULT_PRINCIPAL` is limited to `A-Z a-z 0-9 . _ @ : + -` because the access-control migration interpolates it into DDL, while this mode binds both values as statement parameters and accepts any non-empty value, including Auth0 subjects such as `auth0|...` and `google-oauth2|...`.

Steps:

1. **Stop the server.** The statement runs while no request can read or write the entries, which is why it leaves the optimistic-concurrency `version` untouched.
2. **Preview** with `--dry-run`, which prints the number of entries owned by `FROM` and changes nothing:

   ```bash
   mcp-context-server-migrate --source-url sqlite:////path/to/db.sqlite --reassign-owner local 'auth0|abc123' --dry-run
   ```

3. **Execute** by re-running without `--dry-run`. The CLI prints the number of entries reassigned; when no entry is owned by `FROM` it reports that and exits successfully.
4. **Restart the server.**

The statement sets `updated_at` on every reassigned entry and keeps the entry's grants: each grant keeps its grantee, its permission and its `granted_by`, which records who created it and can name the previous owner. On PostgreSQL each grant also records its entry's owner in `entry_owner_id`, and a foreign key carries the reassignment into that column. `FROM` and `TO` must both be non-empty and must differ; otherwise the CLI exits with status 1 and writes nothing. `--reassign-owner` is mutually exclusive with `--compress`, `--decompress` and `--re-embed`, and a co-passed `--embed-missing` does not run: run it separately.

## PostgreSQL Row-Level Security and the CLI

Every PostgreSQL mode of `mcp-context-server-migrate` must connect with the credentials of the role that owns the tables: the source and target of the UUIDv7 migration, `--compress`, `--decompress`, `--re-embed`, `--embed-missing`, `--reassign-owner` and `--disable-rls`. The credentials in `--source-url` and `--target-url` are used as given; `POSTGRESQL_MIGRATION_USER` and `POSTGRESQL_MIGRATION_PASSWORD` do not apply to them. With [row-level security](database-backends.md#row-level-security-postgresql) enabled, the request login role sees only the rows its identity may read, so a mode running as that role would copy, compress, re-embed or reassign a partial set of rows.

Each mode therefore checks, before it reads any data, whether row-level security filters its role on any of the server's tables, and refuses with exit status 1 when it does:

```text
This command must connect with the table owner's credentials: row-level security filters mcp_app on context_entries, tags. Rerun with the owner's connection string (docs/migration-v2-to-v3.md).
```

A statement the database refuses with insufficient privilege (SQLSTATE 42501) at any later point ends the run with the same remediation and exit status 1. Every PostgreSQL connection the CLI opens also sets `row_security = off`, so a read that row-level security would filter raises an error instead of returning partial results. That setting is a session `SET`, so run the CLI through a Direct connection or a session-mode pooler endpoint rather than a transaction-mode pooler.

The owner reads every row because row-level security is enabled without FORCE. `--reassign-owner` binds no caller identity, which the trigger guarding `owner_id` lets through, so it can rewrite owners with row-level security on. The UUIDv7 migration initializes a PostgreSQL target without row-level security, and `--compress` and `--decompress` replace a vector table with one that carries no policies; in each case the next server start with `ENABLE_POSTGRESQL_RLS=true` installs the missing policies and grants.

### Removing Row-Level Security (`--disable-rls`)

`--disable-rls` removes the row-level security the server installed: the read view and its write trigger, every policy of the server, the guard trigger and the helper functions. It disables row-level security only on the tables the server enabled it on, and reports each table it leaves enabled, such as a table whose row-level security was on before the server first installed its policies. Like the other in-place modes it operates on `--source-url` and ignores `--target-url`.

1. **Stop every server instance**, and change the configuration that would start one again to `ENABLE_POSTGRESQL_RLS=false`. An instance started with the flag on installs the objects again; one started with the flag off before the removal refuses to start. The ordered procedure for Helm, Docker Compose and STDIO clients is in [Disabling row-level security](database-backends.md#disabling-row-level-security).
2. **Preview** with `--dry-run`, which prints the statements and the tables it would disable or leave enabled, and changes nothing:

   ```bash
   mcp-context-server-migrate --source-url postgresql://owner:password@host:5432/mcp_context --disable-rls --dry-run
   ```

3. **Execute** by re-running without `--dry-run`. The removal runs as one transaction under the server's schema-initialization lock. When nothing is installed, it reports that and exits successfully.
4. **Start the server** with `ENABLE_POSTGRESQL_RLS=false`.

Set `POSTGRESQL_SCHEMA` for the CLI when the tables live outside `public`. The removal refuses, with exit status 1 and the list of sessions, while any session of a member of the policy role is connected to the database, a dry run included. Superuser sessions do not count, because the server refuses to start as a superuser with row-level security on. `--force`, valid only with `--disable-rls`, removes the objects anyway and logs those sessions. Exit status is 0 on success, 1 for a refusal, and 2 for an unexpected failure. `--disable-rls` is mutually exclusive with `--compress`, `--decompress`, `--re-embed` and `--reassign-owner`, and a co-passed `--embed-missing` does not run.

## End-to-End Checklist: Supabase v2 to v3 With Compression

This is the recommended order for moving an existing v2 deployment to a v3.0.0 Supabase (PostgreSQL) database with TurboQuant compression enabled. The key ordering rule is that compression is always the LAST step: the migration auto-initializes the target with the fp32 layout, and `--compress` converts it afterward.

1. **Provision the empty target database** in Supabase and **enable the `pgvector` extension** (Dashboard -> Database -> Extensions -> vector). The CLI does not run `CREATE DATABASE`, and managed services restrict `CREATE EXTENSION`.
2. **Configure the PostgreSQL connection** for the CLI and server: set `STORAGE_BACKEND=postgresql` and `POSTGRESQL_CONNECTION_STRING`. Use the Direct connection or Session Pooler (port 5432); if you must use the Transaction Pooler (port 6543), also set `POSTGRESQL_STATEMENT_CACHE_SIZE=0`. Set `POSTGRESQL_SCHEMA` to the schema the server will use. Match `EMBEDDING_MODEL` / `EMBEDDING_DIM` to the source data.
3. **Keep compression OFF for the migration**: leave `ENABLE_EMBEDDING_COMPRESSION=false` (or unset) so the target is auto-initialized with the fp32 vector layout the migration copies into.
4. **Run the v2-to-v3 migration** (preview first): `mcp-context-server-migrate --source-url <v2-url> --target-url <v3-supabase-url> --dry-run`, then without `--dry-run`. A same-backend (PostgreSQL -> PostgreSQL) migration copies the embeddings; a cross-backend migration (SQLite -> Supabase) drops them.
5. **Restore embeddings if they were dropped** (cross-backend only): run `mcp-context-server-migrate --source-url <v3-supabase-url> --embed-missing` to regenerate them with the configured provider. (If you also intend to switch models, run `--re-embed` instead.)
6. **Enable compression as the final step**: export `ENABLE_EMBEDDING_COMPRESSION=true` and the `COMPRESSION_*` values, then run `mcp-context-server-migrate --source-url <v3-supabase-url> --compress --dry-run` followed by `--compress`. In multi-pod Kubernetes deployments every pod MUST inherit the same `COMPRESSION_SEED`.
7. **Launch the v3.0.0 server** with `ENABLE_EMBEDDING_COMPRESSION=true` and the same `COMPRESSION_SEED` / `COMPRESSION_BITS` / `COMPRESSION_VARIANT`. The server reads the seed from the singleton `compression_metadata` row; a mismatch raises `ConfigurationError` (exit 78) and refuses to start.

## Reverting

If the migration outcome is unsatisfactory, the source database (untouched by the CLI's read-only connection) remains intact. Point the previous server version at the source database to revert. The target database can be deleted.

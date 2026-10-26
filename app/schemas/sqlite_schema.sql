-- Main context storage table
-- NOTE on the dual key design (SQLite ONLY):
--   - `rowid_int` is the PRIVATE INTEGER PRIMARY KEY AUTOINCREMENT (the SQLite
--     rowid alias). It is the FTS5 external-content rowid AND the AUTOINCREMENT
--     monotonic counter. It is NEVER exposed via MCP tool boundaries.
--     The name is intentionally `rowid_int` (not `rowid`) because SQLite's
--     `ROWID`/`rowid` are reserved implicit column names; a user-defined
--     column named `rowid` would shadow the implicit one.
--   - `id` is the PUBLIC UUIDv7 hex string (32 lowercase hex chars). It is the
--     externally-visible primary key (via the UNIQUE constraint) and the FK
--     target for `tags`, `image_attachments`, `embedding_metadata`, and
--     `embedding_chunks`. PostgreSQL has no equivalent surrogate (its UUID
--     column is the true PRIMARY KEY).
CREATE TABLE IF NOT EXISTS context_entries (
    rowid_int INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    -- Monotonic optimistic-concurrency token. Bumped on every text/metadata
    -- update (see ContextRepository.update_context_entry); a conditional
    -- `WHERE id = ? AND version = ?` makes concurrent same-entry updates
    -- last-writer-by-submission instead of last-writer-by-completion, so an
    -- older-text update can never silently overwrite a newer one (and its
    -- index_tree node rows can never describe stale text).
    version INTEGER NOT NULL DEFAULT 0,
    -- Access-control columns. owner_id is the server-stamped principal that
    -- created the row (NEVER a tool parameter; the application always supplies
    -- it explicitly on INSERT, so the column carries no default here).
    -- visibility decides who may read the row: 'private' is readable by the
    -- owner and every grantee (user or group rows in context_entry_grants),
    -- 'public' by everyone. A write grant authorizes content edits under
    -- either value; visibility changes and deletes are owner-only.
    owner_id TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'public')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);
-- Deduplication lookup index (mirrors apply_content_hash_migration). Kept in the
-- base schema so every initialization path -- server startup AND the migration CLI
-- target init -- provisions it from inception, not only on a later server start.
CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash ON context_entries(thread_id, source, content_hash);
-- The access-control lookup indexes (idx_context_owner, idx_context_owner_thread,
-- idx_context_public) and the covering indexes for scoped reads
-- (idx_context_access_thread, idx_context_access_source, idx_context_access_id)
-- are NOT declared here: this script also runs against EXISTING databases whose
-- context_entries predates owner_id/visibility, where an index on those columns
-- would crash initialization before the column migration can run.
-- apply_access_control_migration creates them right after adding the columns,
-- unconditionally at every startup, so fresh databases, upgraded databases and
-- migration-CLI targets all get them on first boot.

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
-- Tags are a SET of labels per entry: the write path normalizes and deduplicates
-- before inserting, and this index is the database-level backstop that keeps a
-- second insertion path (or a defect in the first) from storing the same label
-- twice, which every reader would then return twice.
CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_entry_tag ON tags(context_entry_id, tag);

CREATE TABLE IF NOT EXISTS image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSON,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);

-- Per-entry access grants. Each row grants one principal ('user') or one
-- group ('group') read or write access to one entry under either visibility;
-- a write grant also grants read. granted_by records the principal that
-- created the grant. The UNIQUE index makes grant insertion idempotent
-- (ON CONFLICT DO NOTHING).
CREATE TABLE IF NOT EXISTS context_entry_grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id TEXT NOT NULL,
    principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK(permission IN ('read', 'write')),
    granted_by TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_grants_entry_principal
ON context_entry_grants(context_entry_id, principal_type, principal_id, permission);
CREATE INDEX IF NOT EXISTS idx_grants_principal
ON context_entry_grants(principal_type, principal_id, context_entry_id);

-- Generation jobs. One row per entry whose generated outputs are outstanding
-- ('pending', or 'running' under a worker's lease) or whose generation gave
-- up ('failed'); an entry without a row has every output it needs. attempts
-- counts the claims of the job, last_error keeps the text of its last
-- failure, claim_token identifies the claim of a running job, created_at
-- orders the queue, and available_at is the time from which a pending job
-- may be claimed or a running job's lease ends, both from the database
-- clock. The queue index serves the scan for the oldest claimable job and
-- the counts per status. apply_generation_jobs_migration runs these two
-- statements on a database that lacks either object.
CREATE TABLE IF NOT EXISTS generation_jobs (
    context_entry_id TEXT PRIMARY KEY NOT NULL REFERENCES context_entries(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claim_token TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_generation_jobs_queue ON generation_jobs (status, available_at, created_at);

-- Metadata field expression indexes are NOT declared here. They are the single
-- responsibility of handle_metadata_indexes (app/migrations/metadata.py), which
-- provisions exactly the set named in METADATA_INDEXED_FIELDS at server startup
-- (the default field set reproduces the former five scalar indexes: status,
-- agent_name, task_name, project, report_type). Keeping them out of the base
-- schema makes the settings-driven sync layer the single source of truth, so a
-- fresh database with a CUSTOM METADATA_INDEXED_FIELDS does not boot with the
-- five defaults that strict-mode reconciliation would then reject as "extra"
-- (and auto-mode would create-then-drop on every fresh init).
--
-- Consequence: a database initialized ONLY by the migration CLI (which does not
-- run handle_metadata_indexes) has no metadata expression indexes until its
-- first server startup provisions them.
--
-- NOTE: 'references' (object) and 'technologies' (array) are NOT indexed in SQLite
-- regardless of configuration. Array/object fields cannot be efficiently indexed
-- with expression indexes in SQLite; queries on these fields use a full table scan
-- with json_each(). For high-performance queries on these fields, use PostgreSQL
-- with the GIN index.

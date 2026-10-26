-- PostgreSQL Schema for MCP Context Server
-- Converted from SQLite schema with PostgreSQL-specific optimizations
-- NOTE: Schema is templated and replaced during initialization (see init_database in app/startup/__init__.py)

-- Function to automatically update updated_at timestamp
-- SET search_path for security (CVE-2018-1058 mitigation)
CREATE OR REPLACE FUNCTION {SCHEMA}.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp;

-- Main context storage table
CREATE TABLE IF NOT EXISTS context_entries (
    id UUID NOT NULL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSONB,
    summary TEXT,
    content_hash TEXT,
    -- Monotonic optimistic-concurrency token. Bumped on every text/metadata
    -- update (see ContextRepository.update_context_entry); a conditional
    -- `WHERE id = ? AND version = ?` makes concurrent same-entry updates
    -- last-writer-by-submission instead of last-writer-by-completion, so an
    -- older-text update can never silently overwrite a newer one (and its
    -- index_tree node rows can never describe stale text).
    version BIGINT NOT NULL DEFAULT 0,
    -- Access-control columns. owner_id is the server-stamped principal that
    -- created the row (NEVER a tool parameter; the application always supplies
    -- it explicitly on INSERT, so the column carries no default here).
    -- visibility decides who may read the row: 'private' is readable by the
    -- owner and every grantee (user or group rows in context_entry_grants),
    -- 'public' by everyone. A write grant authorizes content edits under
    -- either value; visibility changes and deletes are owner-only.
    owner_id TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'public')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to auto-update updated_at on row modification
DROP TRIGGER IF EXISTS update_context_entries_updated_at ON context_entries;
CREATE TRIGGER update_context_entries_updated_at
    BEFORE UPDATE ON context_entries
    FOR EACH ROW
    EXECUTE FUNCTION {SCHEMA}.update_updated_at_column();

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);
-- Deduplication lookup index (mirrors apply_content_hash_migration). Kept in the
-- base schema so every initialization path -- server startup AND the migration CLI
-- target init -- provisions it from inception, not only on a later server start.
CREATE INDEX IF NOT EXISTS idx_context_entries_dedup_hash ON context_entries(thread_id, source, content_hash);
-- The access-control lookup indexes (idx_context_owner, idx_context_owner_thread,
-- idx_context_public) are NOT declared here: this script also runs against
-- EXISTING databases whose context_entries predates owner_id/visibility, where an
-- index on those columns would crash initialization before the column migration
-- can run. apply_access_control_migration creates them right after adding the
-- columns, unconditionally at every startup, so fresh and upgraded databases both
-- get them on first boot.

-- Tags table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS tags (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
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

-- Image attachments table
CREATE TABLE IF NOT EXISTS image_attachments (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
    image_data BYTEA NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSONB,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);

-- Per-entry access grants. Each row grants one principal ('user') or one
-- group ('group') read or write access to one entry under either visibility;
-- a write grant also grants read. granted_by records the principal that
-- created the grant. The UNIQUE index makes grant insertion idempotent
-- (ON CONFLICT DO NOTHING).
CREATE TABLE IF NOT EXISTS context_entry_grants (
    id BIGSERIAL PRIMARY KEY,
    context_entry_id UUID NOT NULL,
    principal_type TEXT NOT NULL CHECK(principal_type IN ('user', 'group')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK(permission IN ('read', 'write')),
    granted_by TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
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
    context_entry_id UUID PRIMARY KEY REFERENCES context_entries(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claim_token TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    available_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generation_jobs_queue ON generation_jobs (status, available_at, created_at);

-- Scalar metadata field expression indexes are NOT declared here. They are the
-- single responsibility of handle_metadata_indexes (app/migrations/metadata.py),
-- which provisions exactly the typed btree indexes named in
-- METADATA_INDEXED_FIELDS at server startup (the default field set reproduces
-- the former five scalar indexes: status, agent_name, task_name, project,
-- report_type). Keeping them out of the base schema makes the settings-driven
-- sync layer the single source of truth, so a fresh database with a CUSTOM
-- METADATA_INDEXED_FIELDS does not boot with the five defaults that strict-mode
-- reconciliation would then reject as "extra" (and auto-mode would
-- create-then-drop on every fresh init).
--
-- Consequence: a database initialized ONLY by the migration CLI (which does not
-- run handle_metadata_indexes) has no scalar metadata expression indexes until
-- its first server startup provisions them. The always-present GIN index below
-- is NOT managed by the sync layer, so it stays declared in the base schema.

-- GIN index for full JSONB search (enables containment queries)
-- This allows efficient queries like: metadata @> '{"key": "value"}'
-- NOTE: 'technologies' (array) and 'references' (object) fields use this GIN index
-- for containment queries like: metadata @> '{"technologies": ["python"]}'
CREATE INDEX IF NOT EXISTS idx_metadata_gin
ON context_entries USING GIN (metadata jsonb_path_ops);

-- Additional composite index for thread-based queries
CREATE INDEX IF NOT EXISTS idx_thread_created
ON context_entries(thread_id, created_at DESC);

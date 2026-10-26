# Database Backends Guide

## Introduction

The MCP Context Server supports multiple database backends, allowing you to choose the right storage solution for your deployment needs. The backend is selected via the `STORAGE_BACKEND` environment variable.

**Supported Backends:**
- **SQLite** (default): Zero-configuration local storage, perfect for single-user deployments
- **PostgreSQL**: High-performance backend for multi-user and high-traffic deployments
- **Supabase**: Fully compatible PostgreSQL cloud database with managed infrastructure

## SQLite (Default)

Zero-configuration local storage, perfect for single-user deployments.

**Features:**
- No installation required - works out of the box
- Production-grade connection pooling and write queue
- WAL mode for better concurrency
- Suitable for single-user and moderate workloads

**Configuration:** No configuration needed - just start the server!

**Knowledge-base sync reads:** the two reads behind `plan_context_sync` and `list_ingest_state` name their index (`INDEXED BY idx_context_ingest`). Once `sqlite_stat1` exists, written by `ANALYZE` or by the `PRAGMA optimize` the server runs when it closes its writer connection, SQLite picks an index from average rows per key, and a corpus thread far larger than the database's average thread makes another index look more selective: a read through it would seek every member's table row instead of answering from the ingest index, and a `list_ingest_state` page would also sort them. Naming the index keeps every sync read on it whatever the statistics say. See the [Knowledge Base Guide](knowledge-base.md#backend-parity).

## PostgreSQL

High-performance backend for multi-user and high-traffic deployments.

**Features:**
- 10x+ write throughput vs SQLite via MVCC
- Native concurrent write support
- JSONB indexing for fast metadata queries
- Production-grade connection pooling with asyncpg
- pgvector extension for semantic search

**Requirements:** PostgreSQL 14 or newer. The startup migrations create extended statistics on an expression, which PostgreSQL 14 introduced: the server keeps them on the scope of the knowledge-base ingest block so the planner can estimate a corpus thread's members (see the [Knowledge Base Guide](knowledge-base.md#backend-parity)). PostgreSQL 13 reached its end of life in November 2025.

### Quick Start with Docker

Running PostgreSQL with pgvector is incredibly simple - just 2 commands:

```bash
# 1. Pull and run PostgreSQL with pgvector (all-in-one)
docker run --name pgvector18 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context \
  -p 5432:5432 \
  -d pgvector/pgvector:pg18-trixie

# 2. Configure the server (minimal setup - just 1 variable)
export STORAGE_BACKEND=postgresql
# Semantic search registers automatically once an embedding provider extra is installed
# (set ENABLE_SEMANTIC_SEARCH=false to disable).
```

**That's it!** The server will automatically:
- Connect to PostgreSQL on startup
- Initialize the schema (creates tables and indexes)
- Enable pgvector extension (comes pre-installed in the Docker image)
- Apply semantic search migration if enabled
- Record on each access grant the owner of its entry (`context_entry_grants.entry_owner_id`, kept exact by a foreign key onto `context_entries (id, owner_id)` that follows an ownership change and removes the grants with their entry); the first start runs this migration in one transaction that briefly takes an exclusive lock on `context_entries` and `context_entry_grants` (about 0.1 s at 50,000 entries and 9,000 grants), so reads and writes of those tables wait for that moment, and later starts take no lock

### Configuration in .mcp.json

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "mcp-context-server"],
      "env": {
        "STORAGE_BACKEND": "postgresql",
        "POSTGRESQL_HOST": "localhost",
        "POSTGRESQL_USER": "postgres",
        "POSTGRESQL_PASSWORD": "postgres",
        "POSTGRESQL_DATABASE": "mcp_context"
      }
    }
  }
}
```

**Note:** PostgreSQL settings are only needed when using PostgreSQL. The server uses SQLite by default if `STORAGE_BACKEND` is not set.

## Access Isolation by Backend

Both backends enforce the same [Access Model](authentication.md#access-model): apart from the deployment-wide figures in the table below, every query a tool runs against stored entries reaches only the entries the calling principal may read, modify or own, and applies that limit ahead of any `LIMIT`, ranking or aggregation. The server's application layer enforces it on both backends. PostgreSQL can enforce it in the database as well: with `ENABLE_POSTGRESQL_RLS=true`, row-level security policies hold every request to the same rules even where one of the server's queries got its access condition wrong (see [Row-level security (PostgreSQL)](#row-level-security-postgresql)). Without it, and always on SQLite, a client that connects to the database directly, outside the server, sees every row.

| Aspect                                          | SQLite                                                                                                                   | PostgreSQL                                                                                                                                                                   |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Where the access rules are enforced             | Application layer (SQLite has no row-level security)                                                                     | Application layer; with `ENABLE_POSTGRESQL_RLS=true`, also the database                                                                                                      |
| A client connecting to the database directly    | Sees every row                                                                                                           | Sees every row with row-level security off; with it on, a role outside the policy role reads nothing, and the request login role reads only what the bound identity may read |
| Results, counts, pages and per-entry statistics | Limited to readable entries                                                                                              | Limited to readable entries                                                                                                                                                  |
| Full-text `fts_score`                           | FTS5 `bm25()` uses whole-index statistics, so entries the caller cannot read shift the scores and order of readable ones | `ts_rank_cd` scores each entry on its own                                                                                                                                    |
| Figures that describe the whole deployment      | `get_statistics` sizes, connection metrics and configuration fields, and the full-text rebuild estimate                  | The same, plus planner row estimates in `explain_query=True` plans                                                                                                           |

Which entries match and how many land on a page are the same on both backends. For deployments with more than one principal (`MCP_AUTH_PROVIDER=jwt`), use PostgreSQL, whose full-text scores depend on nothing the caller cannot read, with row-level security enabled.

## Deferred Generation Jobs

A store or update made with `defer_generation=true` commits the entry together with one row in `generation_jobs`, and the server's generation worker generates the entry's embeddings, summary and node summaries afterwards (see [Deferred Generation Settings](environment-variables.md#deferred-generation-settings)). Both base schemas declare the table and its index beside `context_entry_grants`:

| Column               | SQLite                                                                               | PostgreSQL                                                                  | Meaning                                                                                         |
|----------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `context_entry_id`   | `TEXT PRIMARY KEY NOT NULL`, referencing `context_entries(id)` `ON DELETE CASCADE`   | `UUID PRIMARY KEY`, referencing `context_entries(id)` `ON DELETE CASCADE`   | One job per entry; deleting the entry deletes its job                                           |
| `status`             | `TEXT NOT NULL DEFAULT 'pending'`, checked to be `pending`, `running` or `failed`    | The same                                                                    | `running` while a worker holds the job, which clients see as `pending`                          |
| `attempts`           | `INTEGER NOT NULL DEFAULT 0`                                                         | The same                                                                    | Attempts spent: a claim spends one, and a graceful stop gives it back                           |
| `last_error`         | `TEXT`                                                                               | The same                                                                    | The text of the last failure, cut to 2,000 characters; no tool returns it                       |
| `claim_token`        | `TEXT`                                                                               | The same                                                                    | The token of the running claim, on which every statement of the worker about the job is keyed   |
| `created_at`         | `TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP`                                       | `TIMESTAMPTZ NOT NULL DEFAULT now()`                                        | Queue order: the oldest job is claimed first, and a job a text change replaces joins the end    |
| `available_at`       | `TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP`                                       | `TIMESTAMPTZ NOT NULL DEFAULT now()`                                        | For a pending job, the time its backoff ends; for a running job, the end of its lease           |

The index `idx_generation_jobs_queue (status, available_at, created_at)` serves the claim. A worker deletes a job in the transaction that writes its outputs, so the table holds only outstanding work, and an entry without a row is `ready`. Job statements take every time they compare or set from the database's clock: `now()` on PostgreSQL, `datetime('now')` on SQLite.

**Claiming.** A worker claims a job with one write that first marks `failed` every running job whose lease ended after its last attempt, then claims the oldest job that is pending, or running under an ended lease, whose `available_at` has passed and that has attempts left, and finally reads the claimed row back by its token:

```sql
UPDATE generation_jobs
SET status = 'running', attempts = attempts + 1, claim_token = :token, available_at = :now + :lease
WHERE context_entry_id = (
    SELECT context_entry_id FROM generation_jobs
    WHERE status IN ('pending', 'running') AND available_at <= :now AND attempts < :max_attempts
    ORDER BY created_at, context_entry_id
    LIMIT 1
    FOR UPDATE SKIP LOCKED  -- PostgreSQL only
)
```

On PostgreSQL, `FOR UPDATE SKIP LOCKED` lets several replicas claim side by side, each passing over a row another one holds. SQLite has no row locks: the claim runs through the server's write queue, and the `UPDATE` takes the database's write lock as it starts, so two processes that share one database file never claim the same job.

**The worker's connection.** On SQLite the worker runs on the process's own backend, so its writes pass through the one write queue. On PostgreSQL with row-level security off it runs on the request connection pool, which binds no identity. With row-level security on, it opens a connection pool of its own at startup, as the table owner with the migration credentials and with the migration connections' session settings, and keeps it open until the server stops. That pool keeps `POSTGRESQL_POOL_MIN` connections open and uses at most one connection per running job plus one for claiming, within `POSTGRESQL_POOL_MAX`; count those connections against the connection limit of the database or of a pooler such as the Supabase Session Pooler (see [Pool Sizing Configuration](#pool-sizing-configuration)). A process that starts no worker, because neither an embedding nor a summary provider is active, opens no such pool.

**Upgrading.** The server runs its base schema on every start, so the first start of a release with deferred generation creates the empty `generation_jobs` table and its index on an existing database, with `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`; no existing row changes, and later starts find both in place. With row-level security on, that start also grants the policy role its privileges on the table and installs the table's policies. An `mcp-context-server-migrate` run against a database no such release has started yet finds no table and therefore no jobs.

## Row-level security (PostgreSQL)

With `ENABLE_POSTGRESQL_RLS=true` (default `false`), PostgreSQL enforces the [Access Model](authentication.md#access-model) itself, in addition to the server's application layer. A request reads only the entries its principal may read, changes text, metadata, tags and images only where it owns the entry or holds a write grant, changes visibility and deletes only its own entries, and publishes as `public` only when `ACCESS_CONTROL_PUBLISH_ROLE` allows it. With the flag off, PostgreSQL runs the same statements as without the feature: no identity statement, no extra transaction and no extra cost.

### How it works

- **Policies.** `context_entries` carries one PERMISSIVE policy per command: SELECT applies the read rule, INSERT requires the stored owner to be the caller, a `public` visibility to be publishable and the caller to hold the write right, which every principal but the read-only one named by `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` holds, UPDATE applies the write rule, DELETE the owner rule. `context_entry_grants` lets a caller read the grants that name it or one of its groups and every grant on an entry it owns, and lets an entry's owner insert and delete grants on it; an insert must also record the caller as the grantor. Each grant records the owner of its entry in `entry_owner_id`, which a foreign key onto `context_entries (id, owner_id)` keeps equal to the entry's owner through every ownership change, so the owner arms compare that column with the caller and none of the grants policies holds a subquery: a grants policy that read `context_entries` would reach its read policy, which reads the grants table again, and PostgreSQL refuses such a policy as infinitely recursive. The child tables (`tags`, `image_attachments`, `embedding_metadata`, `vec_context_embeddings`, `vec_context_embeddings_compressed`, `context_index_nodes`, `generation_jobs`) allow a read where the parent entry is readable, and an insert or delete where the caller may write the parent. No request updates a child row or a grant, so neither has an UPDATE policy; the generation worker updates `generation_jobs` rows as the table owner. The policy expressions come from the same composer that builds the server's own access condition, so the two cannot drift apart.
- **Guard trigger.** A policy cannot compare a row with its previous version, so a `BEFORE UPDATE` trigger on `context_entries` (`guard_context_entry_access_columns`) keeps `owner_id` immutable, lets only the owner change `visibility`, and refuses a change to `public` unless the caller may publish.
- **Policy role.** Every policy applies to one NOLOGIN role, `POSTGRESQL_RLS_POLICY_ROLE` (default `mcp_context_request`), and never to `PUBLIC`. The login role that serves requests is a member of it. Every other role that neither owns the tables nor bypasses row-level security (as a superuser or a `BYPASSRLS` role does) reads nothing from them, whatever privileges it holds.
- **Identity.** The first statement of every transaction on the request connection binds the caller with one `set_config` call: `app.principal_id`, `app.groups` (an exact `text[]` literal), `app.may_publish` and `app.may_write`, all transaction-local, together with `jit = off` for the transaction. Reads run inside such a transaction too. Four helper functions in the schema (`mcp_context_principal_id()`, `mcp_context_groups()`, `mcp_context_may_publish()`, `mcp_context_may_write()`) hand the values to the policies and the read view and fail closed: an unset or reset identity matches no owner and no grant, so only `public` entries stay readable, and it holds neither the publish right nor the write right. PostgreSQL inlines them into every condition that calls them, so a condition evaluated per row calls no function, and the server creates them with the names in their bodies bound to `pg_catalog`, so no `search_path` a session sets changes what they return.
- **Read view.** A policy's condition is a security barrier: PostgreSQL may use an operator from the query as an index condition only when the operator is marked LEAKPROOF, and the full-text match, the JSON accessors and the numeric casts are not (only a superuser can mark them). Under plain policies the full-text GIN index and the metadata expression indexes would go unused. Plain reads of `context_entries` therefore name the view `context_entries_readable`, which the table owner owns and which carries neither `security_barrier` nor `security_invoker`. Its condition is the read rule of the SELECT policy with a role gate, `pg_has_role(current_user, '<policy role>', 'USAGE')`, folded into the rule's identity values and its `public` arm, and PostgreSQL evaluates each of them once per statement: for a role outside the policy role the identity is empty and the `public` arm is false, so such a role reads nothing. A gate in a condition of its own would sit above every join with the view and keep PostgreSQL from memoizing the view side of the join, so the tag statistics would lose their tag-ordered plan. Because the table owner reads its tables without policies, a query through the view keeps every index the same query uses without row-level security. SELECT on the view is granted to the policy role only, and an `INSTEAD OF` trigger (`reject_writes`) keeps the view read-only. Writes, locking reads and reads of the child tables name the base tables and stay under the policies.
- **Application condition kept.** Every statement keeps the server's own access condition beside the policies and the view, so both layers apply.
- **Total conversions.** Inside a read through the view, the server's own conditions, such as a metadata filter, may be evaluated on rows the caller cannot read. Every conversion of a stored JSON value to a non-text type in the server's SQL is total: a value of another type yields NULL instead of raising an error, so no condition can fail on, or reveal through an error, a value the caller cannot read.
- **Generation worker.** The worker that runs deferred generation jobs connects as the table owner, which the policies do not filter, and writes the embeddings, summaries and node summaries of every owner's entries (see [Deferred Generation Jobs](#deferred-generation-jobs)). A job is authorized when the write that enqueues it runs: the insert policy of `generation_jobs` admits it only for an entry the caller may write, and the worker writes outputs generated from that entry's own stored text, only while the entry still holds that text.
- **Outside row-level security.** `compression_metadata` holds deployment provenance and stays readable to the policy role. The server grants the policy role exactly what requests use: USAGE on the schema; SELECT, INSERT, UPDATE and DELETE on the managed tables; SELECT on `compression_metadata`; USAGE and SELECT on the sequences those tables own; and EXECUTE on `jsonb_merge_patch`. It grants nothing on other tables or sequences of the schema, and no TRUNCATE.

### Roles and credentials

Row-level security needs three roles:

| Role                    | Configured by                                                | What it does                                                                                                                                                                                                                                                                                                              |
|-------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Table owner (migration) | `POSTGRESQL_MIGRATION_USER`, `POSTGRESQL_MIGRATION_PASSWORD` | Owns the tables and runs schema setup, migrations, the privilege grants and the startup checks on a short-lived connection, and the generation worker's writes on a connection pool kept open while an embedding or summary provider is active. Row-level security is enabled without FORCE, so the owner reads every row |
| Policy role (NOLOGIN)   | `POSTGRESQL_RLS_POLICY_ROLE` (default `mcp_context_request`) | The only role the policies name; receives the table privileges                                                                                                                                                                                                                                                            |
| Request login role      | The user of the request connection settings                  | Serves every tool call. Must not own the tables, must be neither superuser nor `BYPASSRLS`, and must be an inheriting member of the policy role                                                                                                                                                                           |

The operator creates the policy role, the login role and the membership; the server never creates or alters roles. On an existing deployment the current user keeps owning the tables and becomes the migration user:

```sql
CREATE ROLE mcp_context_request NOLOGIN;
CREATE ROLE mcp_app LOGIN PASSWORD 'choose-a-password' NOSUPERUSER NOBYPASSRLS;
GRANT mcp_context_request TO mcp_app;
```

| Variable                        | Default               | Purpose                                                     |
|---------------------------------|-----------------------|-------------------------------------------------------------|
| `ENABLE_POSTGRESQL_RLS`         | `false`               | Install the policies and the read view and enforce them     |
| `POSTGRESQL_RLS_POLICY_ROLE`    | `mcp_context_request` | The NOLOGIN role the policies name                          |
| `POSTGRESQL_MIGRATION_USER`     | _(none)_              | The table owner; required when `ENABLE_POSTGRESQL_RLS=true` |
| `POSTGRESQL_MIGRATION_PASSWORD` | _(none)_              | The table owner's password                                  |

The migration connection uses the request connection settings, the `POSTGRESQL_CONNECTION_STRING` DSN when it is set and the individual `POSTGRESQL_*` fields otherwise, with `POSTGRESQL_MIGRATION_USER` and `POSTGRESQL_MIGRATION_PASSWORD` replacing the user and the password, each only when it is set. This holds for the DSN form too: the DSN names the request login role, and the migration variables override its user and password for the migration connection. Host, port, database, schema and SSL settings are shared.

The migration user can also be set with `ENABLE_POSTGRESQL_RLS=false`. The server then migrates as the table owner, serves requests as the login role and grants the policy role its privileges on every start, so the policy role and the membership must exist in that setup as well.

### Enabling row-level security

1. Keep the current user, which owns the tables, as `POSTGRESQL_MIGRATION_USER` and its password as `POSTGRESQL_MIGRATION_PASSWORD`. Nothing changes ownership.
2. As a role administrator, create the policy role and a login role that is its member (the SQL above).
3. Point the request connection settings at the login role: `POSTGRESQL_USER` and `POSTGRESQL_PASSWORD`, or the user and password in `POSTGRESQL_CONNECTION_STRING`.
4. Set `ENABLE_POSTGRESQL_RLS=true` and restart.

The first start grants the policy role its privileges, then installs the helper functions, the guard trigger, the policies and the read view and enables row-level security on every managed table, all as the table owner. The grants and the row-level security objects are two transactions, and each takes the schema-initialization lock only when something in it is missing or differs. Every installed object carries a fingerprint comment, so a later start that finds everything current issues no DDL and takes no lock, and several instances can restart side by side. An object changed outside the server is re-created at the next start with a WARNING naming it. A table that appears later, such as `context_index_nodes` when node summaries are switched on or a vector table re-created by `--compress` or `--decompress`, gets its policies and grants at the next start. Changing `POSTGRESQL_RLS_POLICY_ROLE` re-creates every policy and the view at the next start.

During a rolling enable, instances still running with the old configuration keep working when they connect as the table owner, which bypasses the policies. An instance that already serves requests as the split request login role with the flag off, for example after [Disabling row-level security](#disabling-row-level-security), reads only `public` entries and can no longer store, update or delete any entry once an instance with the flag on has started, because the policies then apply to its login role while it binds no caller identity. Stop every such instance before the first instance with the flag on starts (scale the Deployment to zero; stop every STDIO client), as the disable procedure does. An instance with the old configuration that restarts is refused (see [Startup checks](#startup-checks)) until its configuration is updated.

Upgrading a deployment with row-level security on to a release that changes the policies or the tables they read is a full stop: stop every instance, then start the new release on all of them. Instances of two releases cannot share the database, because each start re-creates the policies the way its own release renders them and refuses to start while a policy its release does not install applies to the login role (see [Startup checks](#startup-checks)). The grant owner column and the grants delete policy are such a change: an older instance refuses to start beside the `context_entry_grants_delete` policy, and while it still runs, its `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups` stores fail, because its grant inserts leave the required `entry_owner_id` column empty. The first start of the new release adds that column under a brief exclusive lock (see [Quick Start with Docker](#quick-start-with-docker)).

The deployment guides show the same steps for [Docker Compose](deployment/docker.md#enabling-row-level-security) and [Helm](deployment/helm.md#row-level-security). For Supabase, see [Row-Level Security on Supabase](#row-level-security-on-supabase).

### Rotating the request login role

Create the new login role, grant it the policy role, point the request connection settings at it and roll the instances: `CREATE ROLE mcp_app_2 LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; GRANT mcp_context_request TO mcp_app_2;`. The policies name the policy role, not the login role, so no DDL runs and no instance is ever left without a policy that applies to it. Revoke the old role's membership once no instance uses it.

### Privileges on the read view

The privileges on `context_entries_readable`, apart from its owner's, must be exactly SELECT for the policy role. A later `GRANT ... ON ALL TABLES IN SCHEMA <schema> TO <role>`, or an `ALTER DEFAULT PRIVILEGES` that covers views, also reaches the view; this is routine on Supabase, where the usual fix for a permission error grants every table in `public` to `postgres`, `anon`, `authenticated` and `service_role`. The next start revokes every such privilege and logs a WARNING naming each revoked grantee and privilege, so no operator step is needed. Until then the role gate keeps every such grantee at zero rows, so the data is never exposed.

### Startup checks

The server refuses to start (exit code 78) with a message naming the cause and the statement that fixes it when:

- `ENABLE_POSTGRESQL_RLS=true` is set with `STORAGE_BACKEND=sqlite`, without `POSTGRESQL_MIGRATION_USER`, or with `POSTGRESQL_MIGRATION_USER` equal to `POSTGRESQL_USER` while `POSTGRESQL_CONNECTION_STRING` is unset;
- the request login role is a superuser or has `BYPASSRLS`, the policy role does not exist, or the login role is not its member (checked before any DDL);
- the request and migration credentials resolve to the same PostgreSQL role. The server compares the `current_user` of both connections, so a pooler that forces a login, a Supabase project-ref user name or an `ALTER ROLE ... SET role` cannot hide it;
- after setup, the login role lacks a privilege the start granted, a managed table does not have row-level security enabled and applying to the login role, a table has FORCE ROW LEVEL SECURITY, a policy the server did not install applies to the login role (through `PUBLIC`, the policy role, the login role or any role it inherits), an installed object differs from what the server renders, or the read view has another owner, `security_barrier` or `security_invoker` set, a missing write trigger, or a grantee other than the policy role;
- `ENABLE_POSTGRESQL_RLS=false` is set while the policies or the read view are installed, because the login role would then read only `public` entries and could no longer store, update or delete any entry. Turn the flag back on, or remove them first (see [Disabling row-level security](#disabling-row-level-security));
- `ENABLE_POSTGRESQL_RLS=false` is set with split roles while a managed table has row-level security enabled by someone else, for example through a dashboard or a restore.

With `MCP_AUTH_PROVIDER=jwt` on PostgreSQL with an HTTP transport and the flag off, the server logs a WARNING at startup: access isolation is then enforced in the application layer only.

### Disabling row-level security

`mcp-context-server-migrate --disable-rls --source-url <table-owner connection>` removes what the server installed: the read view and its write trigger, every policy of the server, the guard trigger and the helper functions. It disables row-level security only on the tables the server enabled it on and reports each table it leaves enabled, such as a table whose row-level security was on before the server first installed its policies. It must connect as the table owner, runs in one transaction under the schema-initialization lock, and refuses (exit 1) while any session of a member of the policy role is connected to the database; `--force` removes the objects anyway and logs those sessions. Superuser sessions, such as a DBA's `psql` or a managed platform's own connections, do not count: the server refuses to start as a superuser with row-level security on, so no such session can be a server instance. `--dry-run` prints what would be removed and changes nothing. Set `POSTGRESQL_SCHEMA` for the CLI when the tables live outside `public`. See [Migrating to the UUIDv7 Schema](migration-v2-to-v3.md#postgresql-row-level-security-and-the-cli) for the CLI's owner requirement.

The removal and the configuration change must happen in order, because an instance started with the flag on installs the objects again, and an instance started with the flag off before the removal is refused:

- **Helm:** set `storage.postgresql.rls.enabled: false` in your values; scale the Deployment to zero so no instance runs; run `--disable-rls` from a one-off pod with the table owner's credentials; then apply the values and scale back up. See [Helm Deployment Guide](deployment/helm.md#row-level-security).
- **Docker Compose:** stop the app (`docker compose stop mcp-context-server`, keeping a bundled PostgreSQL sidecar running; `docker compose down` with an external database), run `--disable-rls`, set `ENABLE_POSTGRESQL_RLS=false` (or drop the override that enables it), then `docker compose up -d`. See [Docker Deployment Guide](deployment/docker.md#enabling-row-level-security).
- **STDIO clients:** change every client configuration to `ENABLE_POSTGRESQL_RLS=false` first, then run `--disable-rls` once no session remains.

The request login role can keep its membership: with the flag off and the roles still split, the server keeps granting the policy role its privileges. To return to a single role, point the request connection settings back at the table owner and unset the migration variables.

### Performance

The figures below come from PostgreSQL 18.6: the median of 25 runs of each statement, against the same statement run by the table owner without row-level security and with `jit` off. A figure is in range when its median stays at or below the 95th percentile of those runs without row-level security. Two kinds of data took part, each with 50,000 entries: entries that belong to several principals, with 10,039 `public` entries and 9,006 grants, read as a principal whose user and group grants number 3,460 (the caller with grants below), and entries that all belong to one owner.

- **Plans.** Reads of `context_entries` through the view keep every index the same read uses with row-level security off, the full-text GIN index and the metadata expression indexes included. The tag statistics, which read `tags` under its policy, also keep every index their plan uses without row-level security.
- **Per request.** Each read on the request connection adds three round trips (`BEGIN`, the identity statement, `COMMIT`), and each write transaction adds one. Over a pooled connection inside the database's Docker network, the three took 0.17 ms per read: 0.31 ms for a wrapped statement against 0.14 ms for a bare one. Deployments with the flag off pay nothing.
- **`jit`.** `jit` is off for each request transaction: the policy-heavy plans gain nothing from compilation, and the setting is transaction-local, so it holds through transaction-mode poolers without an extra round trip. With PostgreSQL's default `jit = on`, a server without row-level security runs the whole-table counts of `get_statistics` and `list_threads` for the caller with grants three to five times as long as with `jit` off (the entry count: 72.0 ms against 15.3 ms), so on a server with default settings these counts run faster with row-level security on (19.1 ms). For a single owner, `jit` made no measurable difference.
- **Pages of entries.** For the caller with grants, a page of entries (a browse page, the duplicate check before a store, a metadata `array_contains` filter, the entries a semantic search returns) takes 1.9 to 2.2 ms longer, and keeping the application condition beside the view accounts for about 1.1 ms of that. A read by id takes 0.02 ms longer. For a single owner, small reads differ by 0.01 to 0.02 ms.
- **Searches and counts.** For the caller with grants, a full-text search takes 1.5% (common term) to 8% (selective term) longer, a metadata `in` filter on an indexed field 10% longer, the fp32 semantic search leg 41% longer, and most whole-table counts of `get_statistics` 10% to 35% (1.5 to 6.2 ms) longer. For a single owner they stay in range, apart from one whole-table count 0.25 ms above it.
- **Tools.** Over the HTTP endpoint, 50 calls per tool by a caller without a token, who reads only the `public` entries of that database, gave these medians with row-level security on and off, the server without it running with PostgreSQL's default `jit`: `get_context_by_ids` 57.7 ms against 53.1 ms, `search_context` 71.1 against 66.5, `navigate_context` 56.2 against 54.7 and `read_context_range` 54.3 against 52.7. For the searches, `store_context` and `update_context`, where embedding and summary generation take most of the time, the medians differed by less than 2%.

| Statement (median ms)                     | Caller with grants, off | Caller with grants, on | Single owner, off | Single owner, on |
|-------------------------------------------|-------------------------|------------------------|-------------------|------------------|
| Browse page (`search_context`)            | 0.755                   | 2.84                   | 0.024             | 0.039            |
| Entries by id (`get_context_by_ids`)      | 0.016                   | 0.032                  | 0.013             | 0.029            |
| Full-text search, selective term          | 25.96                   | 28.11                  | 41.75             | 41.79            |
| Full-text search, common term             | 143.30                  | 145.46                 | 230.94            | 233.48           |
| Metadata `in` filter on an indexed field  | 8.53                    | 9.43                   | 2.61              | 3.07             |
| Semantic search candidates (compressed)   | 15.21                   | 18.25                  | 13.72             | 14.44            |
| Semantic search payload read (compressed) | 9.91                    | 39.95                  | 10.28             | 33.09            |
| Semantic search leg (fp32)                | 148.26                  | 209.53                 | -                 | -                |
| Entry count (`get_statistics`)            | 15.25                   | 19.12                  | 13.96             | 15.29            |
| Embedding counts (`get_statistics`)       | 23.76, 23.94            | 57.11, 60.69           | 30.25, 29.74      | 54.61, 49.08     |
| Tag count, top tags (`get_statistics`)    | 188.05, 185.91          | 194.41, 194.13         | 70.80, 69.69      | 99.97, 261.55    |

The caller-with-grants column comes from a database that holds those entries plus 15 more of the caller's, which carry 250,000 tags (474,834 tags in all), except two rows that come from a database holding only the 50,000 entries, with their 224,834 tags and fp32 embeddings: the fp32 search leg and the tag statistics. [Limits](#limits) covers the payload read, the embedding counts and the tag statistics.

### Limits

Row-level security protects against defects in how this server builds its queries: a missing or wrong access condition, a future repository method that forgets its scope, or a statement that names the wrong relation. It does not protect against a compromised server process or injected SQL. The server asserts the caller's identity per transaction through `set_config`, as PostgREST and Supabase do with `request.jwt.claims`, and anything that can run arbitrary SQL on the request connection can set those values too.

- **No leakproof barrier inside view reads.** PostgreSQL's leakproof protection does not apply to reads through the view, so the server's own conditions may run on index entries and rows the caller cannot read, as they do with row-level security off. No condition the server generates can raise an error on any stored value, and tests pin that, so a regression is caught by the test suite rather than by the database.
- **A timing channel.** By timing many requests, a principal can roughly estimate how many entries it cannot read contain a chosen word or metadata value. The channel is coarse, noisy and content-dependent, exists equally with row-level security off, and lies outside what even a `security_barrier` view guarantees (PostgreSQL's documentation on rules and privileges excludes run-time inference).
- **Deployment-wide figures and planner estimates.** The figures listed in [Access Isolation by Backend](#access-isolation-by-backend) stay global, and `explain_query=True` plans carry row estimates from table-wide statistics.
- **Foreign-key checks.** PostgreSQL checks foreign keys without applying policies. Entry ids are generated by the server as UUIDv7 values, and the child insert policies reject an entry the caller cannot read with the same error as an entry that does not exist, so a foreign-key check reveals nothing.
- **Concurrent grant changes.** Each statement evaluates grants as of its own snapshot (READ COMMITTED), so a statement that runs while a grant is added or removed sees the grants as they were when it started.

These measured costs stay out of the range defined under [Performance](#performance):

- **Reads of many child rows.** Each child table's policy checks the parent entry of every row it returns. At 50,000 entries, reading the compressed embeddings of a semantic search's candidates takes 3.2 times as long for a single owner (33.1 ms against 10.3 ms) and 4.0 times as long for the caller with grants (40.0 ms against 9.9 ms), and the embedding counts of `get_statistics` take 1.7 to 2.5 times as long. Reads of a few child rows, such as one entry's tags or the image count of an empty table, take about 0.03 ms longer.
- **Tag statistics.** The tag count and top tags of `get_statistics` read `tags` under its policy. On data spread over several principals they stay in range or exceed it by 1.4 to 8.3 ms: on 224,834 tags the caller with grants took 194 ms for each against 188 and 186 ms without row-level security, and a caller who reads only `public` entries 78 ms against 95 ms; with 15 more entries of the caller with grants carrying 250,000 tags, they took 283 and 301 ms against 274 and 287 ms for that caller, and 133 and 135 ms against 135 and 134 ms for the caller who reads only `public` entries. On a database whose 50,000 entries belong to one owner they take 100 ms for the count and 262 ms for the top tags against 71 and 70 ms, because all 224,950 tag rows are readable and the policy checks the parent entry of each.
- **PostgreSQL 17.** The other figures in this section and under Performance come from PostgreSQL 18.6. On PostgreSQL 17.11 the tag statistics stay out of range on both kinds of data: with 474,834 tags the caller with grants took 288 and 310 ms against 230 and 239 ms without row-level security, and the single owner 101 and 100 ms against 66 ms.

For deployments with more than one principal, use PostgreSQL with row-level security enabled.

### Backups

Run `pg_dump` as the table owner or a superuser. As the request login role it fails under the policies.

## External Connection Pooler Compatibility

When using external PostgreSQL connection poolers (PgBouncer in transaction mode, Pgpool-II, AWS RDS Proxy, etc.), you may encounter connection errors caused by asyncpg's prepared statement caching.

### The Problem

asyncpg (the PostgreSQL driver used by this server) caches prepared statements on the backend connection. When an external pooler reassigns backend connections between requests:

1. The client sends a query referencing a cached prepared statement
2. The backend connection has been reassigned - the statement does not exist on this connection
3. Result: `"connection was closed in the middle of operation"` or similar errors

### The Solution

Disable prepared statement caching by setting:

```bash
POSTGRESQL_STATEMENT_CACHE_SIZE=0
```

### Configuration Reference

| Environment Variable                           | Default   | Description                                                                                       |
|------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------|
| `POSTGRESQL_STATEMENT_CACHE_SIZE`              | 100       | Prepared statement cache size. Set to `0` to disable caching for external pooler compatibility.   |
| `POSTGRESQL_MAX_CACHED_STATEMENT_LIFETIME_S`   | 300       | Maximum lifetime of cached statements in seconds. Has no effect when cache size is 0.             |
| `POSTGRESQL_MAX_CACHEABLE_STATEMENT_SIZE`      | 15360     | Maximum statement size to cache in bytes (15KB). Has no effect when cache size is 0.              |

### Pooler-Specific Notes

**Works without modification (cache enabled):**
- Direct PostgreSQL connection (no pooler)
- PgBouncer in **session mode** (1:1 client-to-backend mapping)
- Supabase **Session Pooler** (maintains session state)

This list is about the STATEMENT CACHE only. Every pooler in both lists is reached with a stock configuration, because the server sends no PostgreSQL startup parameters at all -- see [Startup Parameters and Pooler Allowlists](#startup-parameters-and-pooler-allowlists).

**Requires `POSTGRESQL_STATEMENT_CACHE_SIZE=0`:**
- PgBouncer in **transaction mode** (reassigns connections between transactions)
- Pgpool-II (all modes - reassigns connections across clients)
- AWS RDS Proxy (transaction-level pooling)
- Supabase **Transaction Pooler** (serverless mode)
- Any pooler that reassigns backend connections

**With [row-level security](#row-level-security-postgresql) on:** the caller's identity and `jit = off` are set transaction-locally inside an explicit transaction, which a transaction-mode pooler keeps on one server connection, so requests are safe through every pooler in both lists. The migration connections and the generation worker's table-owner connections set `row_security = off` with a session `SET`, like `search_path`; run the `mcp-context-server-migrate` CLI through a Direct connection or a session-mode endpoint.

### Example Configuration

For environments using external connection poolers:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "mcp-context-server"],
      "env": {
        "STORAGE_BACKEND": "postgresql",
        "POSTGRESQL_HOST": "your-pooler-host",
        "POSTGRESQL_USER": "postgres",
        "POSTGRESQL_PASSWORD": "your-password",
        "POSTGRESQL_DATABASE": "mcp_context",
        "POSTGRESQL_STATEMENT_CACHE_SIZE": "0"
      }
    }
  }
}
```

### Startup Parameters and Pooler Allowlists

The server sends **no PostgreSQL startup-packet parameters**. Every session parameter it depends on is applied with `SET` on each connection instead:

| Session parameter                              | Value                           | Purpose                                                                                                                                                                                                              |
|------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `search_path`                                  | `"<POSTGRESQL_SCHEMA>", public` | Makes bare table names resolve to the configured schema                                                                                                                                                              |
| `extra_float_digits`                           | `1`                             | Pins float text output to the shortest round-trip form the numeric metadata-filter comparison depends on                                                                                                             |
| `tcp_keepalives_idle` / `_interval` / `_count` | `POSTGRESQL_TCP_KEEPALIVES_*`   | Server-side keepalive probes; see [Connection Resilience](#connection-resilience)                                                                                                                                    |
| `row_security`                                 | `off`                           | Migration connections, the generation worker's connections with row-level security on, and the migration CLI only: a read that row-level security would filter raises an error instead of returning a partial answer |

This matters because PgBouncer accepts only `client_encoding`, `datestyle`, `timezone`, `standard_conforming_strings` and `application_name` in the startup packet by default, and **refuses** the connection for anything else with `FATAL: unsupported startup parameter: <name>`. Sending even one non-allowlisted parameter would make a stock `pgbouncer.ini` reject every connection this server opens, before any query runs. Poolers that silently strip unknown startup parameters instead of rejecting them produce a subtler failure: the connection succeeds and the parameter is simply gone, so with a non-default `POSTGRESQL_SCHEMA` every query fails with `relation "context_entries" does not exist`, and float text reverts to the cluster default, which the numeric metadata-filter comparison relies on being pinned.

Applying the parameters as statements avoids both. **No `ignore_startup_parameters` or `track_extra_parameters` entry is required for this server.**

The pool re-applies them on every acquire, which is what makes the statement form safe: releasing a connection issues `RESET ALL`, and the next caller receives it configured again before its first query. One-off connections (the boot-time provisioning probes and both migration CLIs) apply the same statements immediately after dialing.

### Diagnostic Steps

If you encounter connection errors with PostgreSQL:

1. **Check the error message**: Look for `"connection was closed in the middle of operation"`, `"prepared statement does not exist"`, `"unsupported startup parameter"`, or similar
2. **Identify your pooler type**: Determine whether you're using an external pooler and its mode
3. **Apply the fix**: Set `POSTGRESQL_STATEMENT_CACHE_SIZE=0` in your environment; for `unsupported startup parameter`, allowlist the two startup parameters in the pooler as shown above
4. **Verify**: Restart the server and confirm operations complete successfully

### Additional Considerations

**Pgpool-II users**: Some Pgpool-II versions prior to 4.2.3 have known issues with asyncpg's extended query protocol. If disabling statement caching does not resolve issues, consider upgrading Pgpool-II.

**Performance impact**: Disabling statement caching has minimal performance impact for this server. Embedding generation (which takes 800ms+) dominates latency; the overhead of re-preparing statements is negligible.

### External References

- [asyncpg FAQ: Using asyncpg with external poolers](https://magicstack.github.io/asyncpg/current/faq.html)
- [asyncpg Issue #309: Connection closed during operation](https://github.com/MagicStack/asyncpg/issues/309)
- [asyncpg Issue #573: Pgpool-II compatibility](https://github.com/MagicStack/asyncpg/issues/573)

## Using with Supabase

Supabase is fully compatible with the PostgreSQL backend using direct database connection. No special configuration needed - Supabase IS PostgreSQL.

### Connection Methods

Supabase offers TWO connection methods. Choose based on your network capabilities:

1. **Direct Connection** (IPv6 required, lowest latency)
2. **Session Pooler** (IPv4 compatible, universal)

### Connection Method 1: Direct Connection (Recommended)

Best for: VMs, servers, and local development with IPv6 support

**Requirements:**
- IPv6 connectivity (or paid dedicated IPv4 add-on)
- Port 5432 accessible
- Lowest latency (~15-20ms)

**Quick Setup:**

1. **Get your database connection details:**

   Navigate to your Supabase Dashboard:
   - Go to **Database -> Settings** (left sidebar -> Database -> Settings)
   - Find the **"Connect to your project"** section
   - Select **"Connection String"** tab, then **"Direct connection"** method
   - You'll see: `postgresql://postgres:[YOUR_PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres`

2. **Get or reset your database password:**

   **IMPORTANT:** For security reasons, your database password is **never displayed** in the Supabase Dashboard.

   You must use one of these approaches:
   - **Use the password you set** when creating your Supabase project, OR
   - **Click "Reset database password"** (below the connection string) to generate a new password

   **Note:** Replace `[YOUR_PASSWORD]` in the connection string with your actual database password (NOT API keys - those are for REST/GraphQL APIs).

3. **Configure the connection:**

   Add to your `.mcp.json` with your actual password:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres:your-actual-password@db.[PROJECT_REF].supabase.co:5432/postgres"
         }
       }
     }
   }
   ```

   Or using individual environment variables:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_HOST": "db.[PROJECT_REF].supabase.co",
           "POSTGRESQL_PORT": "5432",
           "POSTGRESQL_USER": "postgres",
           "POSTGRESQL_PASSWORD": "your-actual-password",
           "POSTGRESQL_DATABASE": "postgres"
         }
       }
     }
   }
   ```

   **Replace** `[PROJECT_REF]` with your actual project reference ID and `your-actual-password` with your database password.

### Connection Method 2: Session Pooler (IPv4 Compatible)

Best for: Systems without IPv6 support (Windows, corporate networks, restricted environments)

**Requirements:**
- IPv4 connectivity (works universally)
- Port 5432 accessible
- Slightly higher latency (~20-30ms, +5-10ms vs Direct)

**Important Differences from Direct Connection:**
- **Different hostname**: Uses `*.pooler.supabase.com` (NOT `db.*.supabase.co`)
- **Different username format**: `postgres.[PROJECT-REF]` (includes project reference)
- **Same port**: 5432 (NOT 6543 - that's Transaction Pooler for serverless only)
- **IPv4 compatible**: Works on all systems without IPv6 configuration

**Quick Setup:**

1. **Get your Session Pooler connection string:**

   Navigate to your Supabase Dashboard:
   - Go to **Database -> Settings** (left sidebar -> Database -> Settings)
   - Find the **"Connect to your project"** section
   - Select **"Connection String"** tab, then **"Session pooler"** method (NOT "Transaction pooler")
   - You'll see: `postgresql://postgres.[PROJECT-REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres`

   **Example:**
   ```text
   postgresql://postgres.abcdefghijklmno:your-password@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```

2. **Get or reset your database password** (same as Direct Connection - see above)

3. **Configure the connection:**

   Add to your `.mcp.json`:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres.[PROJECT-REF]:your-actual-password@aws-0-[REGION].pooler.supabase.com:5432/postgres"
         }
       }
     }
   }
   ```

   Or using individual environment variables:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_HOST": "aws-0-[REGION].pooler.supabase.com",
           "POSTGRESQL_PORT": "5432",
           "POSTGRESQL_USER": "postgres.[PROJECT-REF]",
           "POSTGRESQL_PASSWORD": "your-actual-password",
           "POSTGRESQL_DATABASE": "postgres"
         }
       }
     }
   }
   ```

   **Replace** `[PROJECT-REF]` with your actual project reference, `[REGION]` with your project region (e.g., `us-east-1`), and `your-actual-password` with your database password.

### Which Connection Method Should I Use?

| Consideration               | Direct Connection               | Session Pooler                 |
|-----------------------------|---------------------------------|--------------------------------|
| **IPv6 Required**           | Yes (or paid IPv4 add-on)       | No - IPv4 compatible           |
| **Latency**                 | Lowest (~15-20ms)               | +5-10ms overhead               |
| **Windows Compatibility**   | May require IPv6 config         | Works universally              |
| **Corporate Networks**      | May be blocked                  | Usually works                  |
| **Configuration**           | Simpler (standard PostgreSQL)   | Requires correct hostname      |
| **Best For**                | VMs, servers with IPv6          | Windows, restricted networks   |

**Recommendation:**
- **Try Direct Connection first** - it's simpler and faster
- **Switch to Session Pooler if you get "getaddrinfo failed" errors** (indicates IPv6 connectivity issues)

### Session Pooler Connection Limits

**IMPORTANT:** Supabase Session Pooler enforces a `pool_size` limit on total concurrent connections. The default application pool settings (`POSTGRESQL_POOL_MAX=20`) may exceed this limit, causing connection errors.

#### Supabase Session Pooler Limits by Tier

| Plan                       | pool_size   | Notes                   |
|----------------------------|-------------|-------------------------|
| Free (Nano)                | 15          | Shared infrastructure   |
| Pro (Micro/Small/Medium)   | 15          | Dedicated compute       |
| Pro (Large+)               | 30+         | Scales with compute     |

**Note:** These are Session Pooler limits. Direct Connection has higher limits (60-500+ depending on tier). Check your Supabase Dashboard -> Settings -> Database for exact values.

#### Pool Sizing Configuration

Configure the application connection pool to fit within Supabase limits:

**Single replica deployment:**

```json
{
  "env": {
    "POSTGRESQL_POOL_MIN": "1",
    "POSTGRESQL_POOL_MAX": "10"
  }
}
```

**Multi-replica deployment (Kubernetes):**

Use this formula to calculate safe pool settings:

```text
POSTGRESQL_POOL_MAX = floor(pool_size / replica_count) - 1
```

| Replicas   | Supabase pool_size   | Recommended POSTGRESQL_POOL_MAX   |
|------------|----------------------|-----------------------------------|
| 1          | 15                   | 10                                |
| 2          | 15                   | 6                                 |
| 3          | 15                   | 4                                 |
| 4          | 15                   | 3                                 |

**Example for 3-replica Kubernetes deployment with Supabase Free tier:**

```json
{
  "env": {
    "POSTGRESQL_POOL_MIN": "1",
    "POSTGRESQL_POOL_MAX": "4"
  }
}
```

**Why leave headroom?** Reserve connections for:
- Administrative queries (Supabase Dashboard)
- The generation worker's table-owner pool on each replica with row-level security on (see [Deferred Generation Jobs](#deferred-generation-jobs))
- Connection spikes during pod restarts
- Other applications sharing the pooler

### Row-Level Security on Supabase

[Row-level security](#row-level-security-postgresql) works on Supabase with dedicated roles:

1. In the SQL Editor, as `postgres`, create the policy role and a login role that is its member: `CREATE ROLE mcp_context_request NOLOGIN; CREATE ROLE mcp_app LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; GRANT mcp_context_request TO mcp_app;`.
2. Keep `postgres`, which owns the tables, as `POSTGRESQL_MIGRATION_USER` with its database password as `POSTGRESQL_MIGRATION_PASSWORD`.
3. Point the request connection at the login role. Through the Session Pooler, user names carry the project reference: `mcp_app.[PROJECT-REF]` for requests and `postgres.[PROJECT-REF]` for migrations. With `POSTGRESQL_CONNECTION_STRING`, put `mcp_app.[PROJECT-REF]` in the DSN and set `POSTGRESQL_MIGRATION_USER=postgres.[PROJECT-REF]`.
4. Set `ENABLE_POSTGRESQL_RLS=true`.

Supabase grants every table and view in `public` to `anon`, `authenticated` and `service_role` by default, and those grants reach the read view as well. Each start revokes them from the view and logs a WARNING naming them (see [Privileges on the read view](#privileges-on-the-read-view)). For the Data API:

- With row-level security on, `anon` and `authenticated` read zero rows through the base tables and the view, whatever they are granted, because no policy names them and the view's role gate excludes them.
- `compression_metadata` stays outside row-level security, so revoke `anon` and `authenticated` on it: `REVOKE ALL ON public.compression_metadata FROM anon, authenticated;`.
- Disabling row-level security restores Data API access to the context tables unless those grants are revoked as well.
- Tables whose row-level security was enabled before the server first installed its policies stay enabled after `--disable-rls`.

### "MaxClientsInSessionMode" Error

**Symptom:**
```text
MaxClientsInSessionMode: max clients reached - in Session mode max clients are limited to pool_size
```

**Cause:** Application pool exceeds Supabase Session Pooler's `pool_size` limit.

**Solution:**

1. **Reduce pool settings:**
   ```bash
   export POSTGRESQL_POOL_MIN=1
   export POSTGRESQL_POOL_MAX=5
   ```

2. **Check replica count:** If running multiple replicas, divide pool size across all replicas using the formula above.

3. **Consider upgrading:** Higher Supabase tiers have larger pool_size limits.

4. **Use Direct Connection:** If your network supports IPv6, Direct Connection has much higher connection limits (60-500+).

**Startup advisory:** When the server connects through a Supabase Session Pooler endpoint (host `*.pooler.supabase.com` on port 5432) and `POSTGRESQL_POOL_MAX` exceeds the conservative default per-session client cap (~15 on Free/Pro tiers), it logs a WARNING at startup naming this symptom and fix. The advisory never clamps the pool; it surfaces the misconfiguration loudly instead of as intermittent search failures.

### Statement Timeout on store/update (SQLSTATE 57014) in fp32 mode

**Symptom:**
```text
canceling statement due to statement timeout
```
emitted by `store_context` / `update_context` (and their batch variants) while PostgreSQL is the backend, or logged by the generation worker as the error of a failed attempt of a deferred job.

**Cause:** Each connection sets `statement_timeout` to `0.9 * POSTGRESQL_COMMAND_TIMEOUT_S` (about 54s with the 60s default). Embedding, summary and compression generation runs outside the transaction that writes its outputs: before the write transaction of a synchronous store or update, and before the generation worker's completion transaction for a write made with `defer_generation=true`. The timeout is therefore hit by the database write itself. The dominant in-transaction cost appears only in fp32 mode (`ENABLE_EMBEDDING_COMPRESSION=false`): every per-chunk INSERT into `vec_context_embeddings` performs an HNSW graph insert inside the transaction that writes the embeddings. Under concurrent writers, the transaction both waits on locks and does slow index maintenance, and can exceed the ceiling.

**Why the default is already mitigated:** With compression ON (`ENABLE_EMBEDDING_COMPRESSION=true`, the default since v3.0.0) the fp32 `vec_context_embeddings` table and its HNSW index are replaced by `vec_context_embeddings_compressed` (a plain `BYTEA` table with no ANN index). Compressed writes are plain INSERTs with no in-transaction HNSW maintenance, so the slow-op cause disappears.

**Solution (in priority order):**

1. **Keep compression ON (recommended default).** This removes HNSW maintenance from the write path entirely and is the structural fix for the slow-op cause. See [`docs/embedding-compression.md`](embedding-compression.md).
2. **If you must run fp32** (`ENABLE_EMBEDDING_COMPRESSION=false`), raise the timeout to fit the slowest expected write under your concurrency, e.g.:
   ```bash
   export POSTGRESQL_COMMAND_TIMEOUT_S=180
   ```
   The per-connection `statement_timeout` follows automatically at `0.9 *` this value.
3. **Reduce write concurrency** if a pooler or replica count is forcing many simultaneous fp32 writes through the HNSW index.

**Automatic retry (transient cases only):** A statement cancellation (SQLSTATE 57014) is retried with bounded exponential backoff at both the tool layer (store/update/batch) and in `execute_write`. Retry uses the SAME timeout ceiling, so it recovers only a TRANSIENT lock-WAIT (the write was blocked behind another writer and the contention has cleared by the retry). It does NOT rescue a write that is fundamentally slower than the ceiling -- for that, apply solution 1 or 2 above. Retry is safe because PostgreSQL aborts the transaction of a cancelled statement, so the failed attempt never committed, and generation already completed outside the transaction, so a retry never regenerates or skips embeddings/summaries.

## Troubleshooting

### "getaddrinfo failed" Error

If you see this error with Direct Connection:
```text
Error: getaddrinfo ENOTFOUND db.[PROJECT-REF].supabase.co
```

**Solution:** Your system doesn't support IPv6 or it's disabled. Use Session Pooler instead (Method 2 above).

**Why?** Direct Connection (`db.*.supabase.co`) uses IPv6 by default. Session Pooler (`*.pooler.supabase.com`) provides IPv4 compatibility through Supavisor proxy.

### Enabling Semantic Search (pgvector extension)

If you want to use semantic search with Supabase, you must enable the pgvector extension:

1. **Via Supabase Dashboard** (easiest method):
   - Navigate to **Database -> Extensions** (left sidebar)
   - Search for "vector" in the extensions list
   - Find "vector" extension (version 0.8.0+)
   - Click the **toggle switch** to enable it (turns green when enabled)

2. **Via SQL Editor** (alternative method):
   - Navigate to **SQL Editor** in Supabase Dashboard
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`
   - Verify: `SELECT * FROM pg_extension WHERE extname = 'vector';`

3. **Configure environment**:
   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres:your-actual-password@db.[PROJECT_REF].supabase.co:5432/postgres"
         }
       }
     }
   }
   ```

**Note:** The pgvector extension is available on all Supabase projects but must be manually enabled. The server will automatically create the necessary vector columns and indexes on first run.

## Connection Resilience

The PostgreSQL backend includes multi-layered protection against connection drops caused by network intermediaries (NAT, firewalls, proxies) that may close idle connections.

### TCP Keepalive (Automatic)

TCP keepalive is enabled by default on all PostgreSQL connections:

| Setting          | Env Variable                             | Default   | Description                               |
|------------------|------------------------------------------|-----------|-------------------------------------------|
| Idle timeout     | `POSTGRESQL_TCP_KEEPALIVES_IDLE_S`       | 15        | Seconds before first keepalive probe      |
| Probe interval   | `POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S`   | 5         | Seconds between keepalive probes          |
| Probe count      | `POSTGRESQL_TCP_KEEPALIVES_COUNT`        | 3         | Failed probes before connection is dead   |

**How it works:**
- **Client-side** (PRIMARY): Configured via `setsockopt()` on each connection's socket. This is the only layer that reaches the server's own socket, and the only one that applies at all when an external pooler terminates the connection.
- **Server-side** (SECONDARY): The matching `tcp_keepalives_idle` / `tcp_keepalives_interval` / `tcp_keepalives_count` GUCs are applied with `SET` when a pooled connection is prepared, so they are also restored after the pool's `RESET ALL`. They are effective for direct connections. Behind a pooler they configure the pooler-to-server hop rather than the application's connection, and a transaction-mode pooler may not preserve them at all; the client-side layer covers that case. They are deliberately not sent as startup parameters, which a pooler would have to allowlist (see [Startup Parameters and Pooler Allowlists](#startup-parameters-and-pooler-allowlists)).

**Cross-platform:** Uses `hasattr()` checks for socket constants. Works on Linux, macOS, and Windows 10 v1703+.

To disable TCP keepalive, set all three values to 0:
```bash
POSTGRESQL_TCP_KEEPALIVES_IDLE_S=0
POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S=0
POSTGRESQL_TCP_KEEPALIVES_COUNT=0
```

### Transaction Heartbeat

Long-running transactions (e.g., storing context with embeddings) include periodic `SELECT 1` heartbeat queries between sequential database operations. This prevents intermediaries from classifying the connection as idle during inter-operation gaps.

### Transaction Retry

If a transaction fails on a transient fault, it is retried up to 2 times with exponential backoff (0.5s, 1.0s); logical errors are raised immediately. The transient faults are a lost connection, a statement timeout, a deadlock and a serialization failure. PostgreSQL aborts the transaction on each of them, so the failed attempt never committed and the retry is safe. The exception is a connection lost while `COMMIT` is in flight: the outcome is unknown, and the retry runs the write again. An update or a delete by id still leaves no more than one attempt's effect, and a single deduplicating store merges into the entry its first attempt stored, but a `deduplicate=false` store, or an atomic batch store of several entries, can store a second copy of an entry. For an entry of a knowledge-base corpus, the next sync plan lists such a copy as a duplicate delete (see [Concurrency and Recovery](knowledge-base.md#concurrency-and-recovery)).

## Why Direct Connection?

- **Recommended by Supabase** for backend services and server-side applications
- **Full PostgreSQL capabilities**: pgvector (available, must be enabled), JSONB, transactions, all extensions
- **Better performance**: Lower latency than REST API, native connection pooling
- **Production-ready**: MVCC for concurrent writes, connection pooling with asyncpg
- **Zero special code**: Uses standard PostgreSQL backend - no Supabase-specific implementation needed

## Important Notes

- Use **database password** from Settings -> Database section
- **NOT API keys** - API keys (including legacy service_role key) are for REST/GraphQL APIs, not direct database connection
- Use **port 5432** for direct connection (recommended for backend services)
- **pgvector extension** is available on all Supabase projects - enable it via Dashboard -> Extensions for semantic search
- **All PostgreSQL features** work identically - JSONB indexing, metadata filtering, transactions

## Security Best Practices

- Store database password in environment variables, not in code
- Use Supabase's connection string format for simplicity
- Enable SSL/TLS by default (handled automatically by Supabase connection)
- For deployments with more than one principal, serve requests through a dedicated login role with [row-level security](#row-level-security-postgresql) enabled and keep the table owner's credentials for migrations and the generation worker

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start

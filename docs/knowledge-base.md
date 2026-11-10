# Knowledge Base

The server stores session context for agents, and it also hosts document corpora such as a handbook, a set of runbooks or a design archive. Agents read a corpus with the same search, navigation and retrieval tools they use for session context. The `mcp-context-ingest` command uploads a corpus from a directory and keeps the stored entries in sync with the files.

## Corpus Conventions

- **One thread per corpus.** A corpus lives in the thread `kb:<corpus>`, where `<corpus>` is a short, stable, lowercase name such as `kb:handbook`. Session threads keep their own ids. The corpus, its thread and the `scope` in each managed entry's ingest block are one and the same value.
- **Managed and unmanaged entries.** An entry the sync writes carries the ingest block described below. A corpus thread may also hold entries without a block, stored by any other client; the sync never plans, updates or deletes them.
- **Source.** Corpus entries use `source='user'`. The value records provenance and changes nothing about reading or syncing.
- **Owner.** Every entry belongs to the principal the client authenticates as: `ACCESS_CONTROL_DEFAULT_PRINCIPAL` over stdio and with the `none` and `simple_token` providers, and the token subject under `jwt`. For a corpus several people maintain, sync as a dedicated service principal.
- **One owner per corpus.** A sync manages only the entries its own principal owns. When a second principal syncs the same thread, it builds its own copy of the corpus, and each principal's plans count the other's readable entries as foreign entries.

## Visibility

`mcp-context-ingest` sends an explicit visibility with every create and every update: `private`, or `public` when the run passes `--visibility public`. The visibility is part of each entry's `meta_digest`, so a run with a different `--visibility` updates every entry of the corpus to the new value, and a later run without the flag makes them `private` again. Only the owner may change an entry's visibility, which never stands in a sync's way, because a sync updates only the entries its own principal owns.

- **`private`** entries are readable by their owner and every grantee. With `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`, the server writes a read grant for each group of the writing principal when it first stores an entry, so the groups of the principal that syncs a corpus under `jwt` can read it. The owner of an entry can grant further users and groups access with `share_context` (see [Sharing a Corpus](#sharing-a-corpus)).
- **`public`** entries are readable by every caller, which suits a corpus everyone who uses the server may read. That includes every holder of an anonymous token when the server hands one out (see [Anonymous Read Access](authentication.md#anonymous-read-access)). When `ACCESS_CONTROL_PUBLISH_ROLE` is set, only a principal holding that role may store or update an entry as `public`; the server refuses every such create and update from any other principal, and the run exits 1.

On a server that several principals use, run PostgreSQL with row-level security, which enforces the same rules in the database as well; see [Row-level security (PostgreSQL)](database-backends.md#row-level-security-postgresql) and the [Access Model](authentication.md#access-model).

## Sharing a Corpus

The owner of a corpus entry, the principal that syncs the corpus, can grant other users and groups `read` or `write` access to it with `share_context`, remove a grant with `unshare_context` and list the grants of an entry with `list_context_grants` (see [Sharing Entries](authentication.md#sharing-entries)). Grants are per entry: no tool shares a whole thread, so sharing a corpus this way takes one call per entry and grantee.

- **A sync leaves the grants of the entries it keeps alone.** Grants are not part of `meta_digest`, so a grant never makes an entry differ from its document, and an update keeps the entry and every grant on it. The sync sends no grant itself: a run changes grants only by deleting entries, whose grants go with them, and by storing entries, for which the server writes the `author_groups` grants.
- **A re-created entry loses its explicit grants.** When a document is renamed or moved, the sync deletes the old entry and stores the document as a new one, and the same happens to a document that disappears and later returns. Grants made with `share_context` are deleted with the old entry. With `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`, the server writes the author-group read grants again when it stores the new entry.
- **A removed author-group grant stays removed.** An update never writes grants, so unsharing an `author_groups` grant from an entry is permanent for that entry.
- **For a whole corpus, use visibility or author groups.** To open a corpus to every caller, sync it with `--visibility public`. To open it to a group, sync it under `jwt` as a service principal that belongs to the group, with `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`: every entry the sync stores then carries a read grant for the group, re-created entries included.
- **Text edits through a write grant last until the next sync.** A principal holding a write grant may edit a corpus entry, and the grant survives every sync, but the owner's next sync restores the document's text when the stored text differs from it (see [Concurrency and Recovery](#concurrency-and-recovery)).

## The Ingest Metadata Block

Every entry the sync manages carries the reserved top-level metadata key `ingest`:

```json
{"ingest": {"scope": "kb:handbook", "source_id": "guides/setup.md", "meta_digest": "<64 lowercase hexadecimal characters>", "synced_at": "2026-10-18T12:00:00Z"}}
```

| Field         | Value                                         | Meaning                                                                     |
|---------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| `scope`       | a string of 1 to 256 characters               | The corpus; a member's `scope` equals its `thread_id`                       |
| `source_id`   | a string of 1 to 256 characters               | The source document inside the corpus                                       |
| `meta_digest` | exactly 64 characters from `0-9a-f`           | The client's digest of everything it controls except the text               |
| `synced_at`   | `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`, ASCII digits | When a sync last wrote the entry, in UTC; it must name a real date and time |

Lengths count Unicode code points, so a character outside the Basic Multilingual Plane counts once. The 256-character bound on `source_id` keeps the corpus index key within PostgreSQL's index size limit.

**Membership.** An entry belongs to the corpus in thread `T` exactly when its block is valid and its `scope` equals `T`. The server decides this each time it reads the block, so a block whose `scope` names another thread makes its entry a member of no corpus.

**Document identity.** `mcp-context-ingest` writes as `source_id` the file's path relative to the corpus root, with `/` separators, in Unicode NFC, case preserved. The server treats `source_id` as an opaque string and compares it code point by code point, with no case folding and no Unicode normalization, on SQLite and PostgreSQL alike; renaming a file only in case therefore deletes one entry and creates another. To read the entry of one document, filter the corpus thread with `metadata={"ingest.source_id": "guides/setup.md"}`, which matches exactly and case-sensitively.

**Digests.** The server stores a text without its leading and trailing whitespace (Python `str.strip()`) and records the SHA-256 hex digest of the stored text's UTF-8 bytes as the entry's content hash. A client computes the same digest over the text it would send, after the same strip, and the sync plan compares the two, so the block carries no text digest. `mcp-context-ingest` computes `meta_digest` as the SHA-256 hex digest of the canonical JSON of `{"metadata": <the entry metadata without ingest>, "tags": <the tags normalized as the server stores them (trimmed, lower-cased, blanks and duplicates dropped), sorted>, "visibility": <the visibility it sends>}`. Canonical JSON sorts object keys at every depth, uses no whitespace between tokens and keeps non-ASCII characters as UTF-8. A change to a file's metadata, to the tags of a run or to the visibility therefore makes the next sync update the entry.

**Validation on every write.** `store_context`, `store_context_batch`, `update_context` and `update_context_batch` check the block before any embedding or summary generation and refuse a malformed one with a message naming the field: a single-entry tool fails the call, a batch fails that entry, and an atomic batch aborts naming the entry. A block holds exactly the four fields. `metadata_patch` accepts `"ingest": null`, which removes the block, or a complete valid block, which replaces it, and refuses a partial block. A full `metadata` replacement keeps plain replacement semantics: a replacement without `ingest` removes the block, the entry stops being managed and no sync deletes it, and the next sync stores the document again as a new entry. `mcp-context-ingest` always writes the complete block.

**Access.** The block never decides who may read or write an entry; ownership, visibility and grants do. A client that writes a wrong digest affects only its own plans, because a sync plans only the entries its principal owns. A principal holding a write grant on an entry can patch the entry's block; that changes no one's access and at most makes the owner's next sync update or store the document again.

**Edits made outside the sync.** The server compares `meta_digest` values and never recomputes them, so a sync cannot see metadata another client changed on a managed entry. Such an edit persists until the source file's metadata or text changes; the update that follows replaces the metadata with the file's.

## Sync Planning

`plan_context_sync(thread_id, manifest, source_id_after=None, source_id_through=None, allow_empty_manifest=False)` compares a manifest of a corpus's source documents with the caller's own managed entries of the thread and answers what the client must store, update and delete to bring the thread in line with the documents. The tool only reads; the client applies the plan through the batch tools.

**Manifest items.** An ordinary item is `{"source_id": ..., "text_digest": ..., "meta_digest": ...}`: `text_digest` is the SHA-256 digest of the text the client would store, computed as described under Digests above, and `meta_digest` is the digest the client writes into the entry's block. A keep item is `{"source_id": ..., "keep": true}` without digests and says that the source exists but the client could not read it in this run. An item accepts no other key, each `source_id` appears at most once, and a manifest holds at most 1,000 items.

**Ranges.** A plan covers the managed entries whose `source_id` lies in `(source_id_after, source_id_through]`, compared code point by code point; an omitted bound leaves that side open, and every manifest item must lie inside the range. A corpus of more than 1,000 documents is planned in slices: sort the manifest by `source_id` and send consecutive slices of at most 1,000 items, the first without a lower bound, each later one with `source_id_after` set to the last `source_id` of the slice before it, and the last without an upper bound. Every stored entry then falls in exactly one slice, and each slice plans deletes only for entries inside its own range.

**The response.**

| Field             | Meaning                                                                                                                                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `thread_id`       | The planned thread                                                                                                                                                                                                      |
| `creates`         | Manifest indices of the ordinary items that have no stored entry                                                                                                                                                        |
| `updates`         | `{index, context_id, change}` for each ordinary item whose stored entry differs: `change` is `text`, `metadata` or `both`                                                                                               |
| `deletes`         | `{context_id, source_id, reason}`, at most 100, ordered by `source_id` and then `context_id`; `reason` is `absent` when the manifest does not name the source and `duplicate` when another entry of the source survives |
| `deletes_total`   | Every planned delete of the range; while it exceeds the listed deletes, delete the listed entries and plan the range again                                                                                              |
| `unchanged_count` | Ordinary items whose stored entry already matches                                                                                                                                                                       |
| `kept_count`      | Keep items                                                                                                                                                                                                              |
| `foreign_count`   | Readable managed entries of the range that another principal owns                                                                                                                                                       |

The plan compares an item's `text_digest` with the stored entry's content hash, counting an entry without a stored hash as changed, and its `meta_digest` with the one in the entry's block.

**Keep items.** A keep item counts as present: the plan creates, updates and deletes nothing for its source, its duplicates included, and adds it to `kept_count`. A file the client cannot read in one run therefore never deletes what the server holds for it. A keep item for a source without a stored entry creates nothing.

**Duplicates.** When several owned entries carry the same `source_id`, the plan keeps one and lists every other as a `duplicate` delete. The survivor is the entry that ranks first by three keys in turn: its content hash equals the item's `text_digest`, its `meta_digest` equals the item's, and the larger `context_id`, which is the newer entry because ids are UUIDv7. The order is total, so every plan over the same stored entries names the same survivor, a dry run names the same deletes as the run that follows it, and an entry whose text already matches survives a newer entry that matches nothing, so that it needs at most a metadata update instead of a regeneration.

**Foreign entries.** A plan manages only entries the caller owns. A managed entry another principal owns and the caller may read is never updated or deleted; it adds to `foreign_count` and nothing else, and the plan treats its source as if no entry existed, so a manifest item for it plans a create of the caller's own copy. An entry the caller may not read is not counted at all.

**Empty manifests.** An empty manifest plans the deletion of every managed entry in the range, so the tool refuses it unless the call passes `allow_empty_manifest=true`.

**Argument errors.** Before it reads anything, the tool refuses a `source_id_after` that does not sort before `source_id_through`, an empty manifest without the opt-in, a repeated `source_id` (the message names up to ten), an item outside the range, more than 1,000 items, a malformed digest, a keep item that carries a digest and an ordinary item that lacks one.

A thread without managed entries plans every ordinary item as a create, which is how a corpus starts. To apply a plan, store its creates with `store_context_batch`, each entry carrying `"deduplicate": false` and the complete ingest block; bring its updates in line with `update_context_batch`, sending the full `metadata` object rather than a `metadata_patch`; delete its listed deletes with `delete_context_batch(context_ids=...)`, never by `thread_ids`, which would also delete the caller's unmanaged entries of the thread; and plan again until no create, update or delete remains. `mcp-context-ingest` does exactly this.

## Listing Ingest State

`list_ingest_state(thread_id, limit=100, cursor=None)` pages through the caller's own managed entries of a thread, ordered by `source_id` and then `context_id` in code-point order, the order the ranges of `plan_context_sync` use. `limit` is 1 to 200. Each entry carries `context_id`, `source_id`, `text_digest` (the entry's stored content hash, null when the entry has none), `meta_digest` and `synced_at`, and the response adds `thread_id`, `count` and `next_cursor`.

Pass a page's `next_cursor` as `cursor` to read the next page. `next_cursor` is null on the last page, including a page that ends exactly at the last entry. Treat a cursor as opaque: the tool refuses a malformed one with `invalid cursor`, and an altered one can only move the window over the caller's own entries. Entries another principal owns are never listed and never take a slot on a page, and an entry whose block is not valid is skipped without stopping the paging.

The list serves inspection: auditing what a sync manages, comparing it with a source tree, or rebuilding a manifest from the server's side.

## Storing Without Deduplication

`store_context` takes a `deduplicate` parameter, and each entry of `store_context_batch` takes a `deduplicate` key; both default to `true`. With `true`, a store whose text equals that of the latest entry the caller may read in the same thread and source updates that entry instead of inserting a copy, when the caller owns the entry and no entry of the other source follows it; this protects against a client that retransmits a message (see [store_context](api-reference.md#store_context)). With `false`, the store inserts one new entry owned by the caller: it looks for no entry to merge into, always applies the sent visibility, and runs every generation step as for any new entry.

A corpus needs `false`. A corpus thread holds entries of a single source, so the check that tells a retransmit from a new conversational turn never applies, and two documents with identical text would merge: the second store would replace the first entry's metadata, ingest block included, and the first document would drop out of the sync state. `mcp-context-ingest` stores every create with `"deduplicate": false`.

Entries of one batch are stored in order, so an entry with `deduplicate` true that follows an identical `deduplicate: false` entry of the same thread and source merges into the entry just inserted. Keep the flag the same within a batch of corpus documents.

Deduplication plays no part in making a re-upload idempotent: an unchanged corpus writes nothing because the plan reports its documents as unchanged. A store whose outcome is unknown, such as a batch call that timed out on the client, is settled by planning again, never by sending it again: a resent `deduplicate: false` store inserts a second copy, which the next plan then lists as a duplicate delete.

## The mcp-context-ingest CLI

`mcp-context-ingest` synchronizes a directory of documents into the thread `kb:<corpus>` of a running server. It is an ordinary MCP client: it speaks MCP over Streamable HTTP, writes as the principal it authenticates as, reads no server configuration and never connects to the database, so access scoping, row-level security and the server's generation rules apply to its writes exactly as to any other client's: by default the server generates each document's embeddings and summaries before the write commits, and with `--defer-generation` it generates them in the background after the write (see [Generation Status](api-reference.md#generation-status)).

### Installation

The command ships with the server package:

```bash
# Run without installing
uvx --from mcp-context-server mcp-context-ingest --help

# Install mcp-context-server, mcp-context-server-migrate and mcp-context-ingest as tools
uv tool install mcp-context-server
```

In a checkout of the repository, run `uv run mcp-context-ingest`. The server's Docker image carries the command as well; see [Running from the Docker Image](#running-from-the-docker-image).

### A First Run

```bash
mcp-context-ingest --server http://localhost:8000/mcp --corpus handbook --source-dir ./handbook --dry-run
mcp-context-ingest --server http://localhost:8000/mcp --corpus handbook --source-dir ./handbook
```

The first command prints what a sync would create, update and delete and writes nothing; the second stores every `*.md` file under `./handbook` in the thread `kb:handbook`. Running the second command again writes nothing until a file changes.

### Flags

| Flag                           | Default   | Meaning                                                                                                                                   |
|--------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `--server URL`                 | required  | The server's Streamable HTTP endpoint, such as `http://localhost:8000/mcp`                                                                |
| `--corpus NAME`                | required  | The corpus: lowercase letters, digits, `.`, `_` and `-`, starting with a letter or digit; its thread is `kb:NAME`, at most 256 characters |
| `--source-dir PATH`            | required  | The corpus root, walked recursively                                                                                                       |
| `--include GLOB`               | `**/*.md` | Select the files matching GLOB; repeatable, and any `--include` replaces the default                                                      |
| `--exclude GLOB`               | none      | Leave out the files matching GLOB; repeatable                                                                                             |
| `--tag TAG`                    | none      | A tag for every document; repeatable                                                                                                      |
| `--metadata-json JSON`         | `{}`      | A JSON object of metadata for every document; it may not set `ingest`                                                                     |
| `--visibility private\|public` | `private` | The visibility sent with every create and update                                                                                          |
| `--token-file PATH`            | none      | Read the bearer token from PATH before every request                                                                                      |
| `--token-env NAME`             | none      | Read the bearer token once from the environment variable NAME                                                                             |
| `--dry-run`                    | off       | Plan and print the plan; write nothing                                                                                                    |
| `--allow-empty-corpus`         | off       | Let a directory without documents delete every managed entry of the corpus                                                                |
| `--report PATH`                | none      | Write a JSON report of the run to PATH                                                                                                    |
| `--batch-size N`               | 10        | The most entries per batch call, 1 to 100                                                                                                 |
| `--max-request-bytes N`        | 3,145,728 | The request budget per call, at most 4,194,304                                                                                            |
| `--call-timeout SECONDS`       | 3600      | How long one call may take                                                                                                                |
| `--concurrency N`              | 1         | The most create and update batch calls in flight at once, 1 to 20; deletes and plans run one at a time                                    |
| `--defer-generation`           | off       | Ask the server to store every create and update at once and generate its embeddings and summaries in the background                       |

`--token-file` and `--token-env` exclude each other. An invalid flag is a usage error and exits 2.

### Server and Token

`--server` names the server's `/mcp` endpoint; the command speaks no other transport. Without a token flag it sends no `Authorization` header, which suits `MCP_AUTH_PROVIDER=none`. With `simple_token` or `jwt`, give it the token in one of two ways:

- **`--token-file PATH`** reads the file before every request, so a refresher such as an identity-provider agent or a scheduled job can replace the token during a long run. When the server answers 401 after accepting the run's earlier requests and the file now holds a different token, the command retries that one call with the new token.
- **`--token-env NAME`** reads the variable once at the start, so a run must finish within the token's lifetime.

The command never accepts a token as an argument and never prints one, and neither its output nor its report carries a token or the userinfo of the server URL. A refused token ends the run with exit 2; when the server accepted the same token earlier in the run, the message says that it expired.

### Corpus Layout

The walk starts at `--source-dir`, descends into every directory, follows no symbolic link and skips every `.git` directory. A file is selected when at least one `--include` pattern matches its path relative to the root, written with `/` separators, and no `--exclude` pattern does. Patterns match case-sensitively on every host: `**` matches any number of path segments (`**/` also matches none, so `**/*.md` selects `README.md` at the root), `*` and `?` never match `/`, and `[...]` classes work as in `fnmatch`. A Windows host and the Linux image therefore select the same files; `**/*.md` does not select `NOTES.MD` on either.

Each selected file is decoded as UTF-8, a byte order mark accepted, with CRLF and lone CR line endings turned into LF and surrounding whitespace removed, which is the form the server stores. Its `source_id` is its root-relative path in Unicode NFC, case preserved, as [The Ingest Metadata Block](#the-ingest-metadata-block) describes.

### Metadata, Tags and Visibility

Every document's metadata starts with a `title`: the text of the file's first level-one ATX heading (`# Title`), or the file name without its extension when it has none. `--metadata-json` adds its keys to every document, its `title` replacing the derived one; it must be a JSON object without an `ingest` key, because the command writes the ingest block itself. `--tag` adds tags to every document, and `--visibility` sets the visibility of every document (see [Visibility](#visibility)). All three feed `meta_digest`, so a run that changes any of them updates every affected entry's metadata, tags and visibility without regenerating its embeddings or summaries.

### What a Run Does

1. **Read the corpus.** Walk the directory, build the documents, and sort their manifest by `source_id` into slices of at most 1,000 items, fewer when a slice's plan request would exceed `--max-request-bytes`.
2. **Preflight.** Check that the server offers `plan_context_sync`, `store_context_batch`, `update_context_batch` and `delete_context_batch`, and that `store_context` has the `deduplicate` parameter; a server without one of them ends the run with exit 2 naming it. A server that does not know `deduplicate` would merge identical documents silently, which this check prevents. With `--defer-generation` the preflight also requires the `defer_generation` parameter of `store_context`, and a server without it ends the run with exit 2 and `the server's store_context has no defer_generation parameter, so it cannot defer generation: upgrade the server or run without --defer-generation`.
3. **Plan.** Plan every slice and print how many documents the run will create, update and delete and how many are unchanged, kept and foreign. A non-zero foreign count prints a warning that another principal owns managed entries of the thread, which the run leaves untouched.
4. **Create.** Store the planned creates with `store_context_batch`, non-atomic, every entry carrying `"deduplicate": false`, the complete ingest block and the explicit visibility, and with `--defer-generation` also `"defer_generation": true`.
5. **Update.** Send the planned updates through `update_context_batch`, non-atomic. A `text` or `both` change sends the text with the full metadata, tags and visibility; a `metadata` change sends the full metadata, tags and visibility without the text. Metadata is always replaced in full, never patched. With `--defer-generation` every update carries `"defer_generation": true`, which the server applies to the updates that send text.
6. **Delete.** Plan every slice again and delete each slice's listed deletes with `delete_context_batch(context_ids=...)`, planning again until `deletes_total` is 0 or a round removes nothing. Deletes run last, so a run that stops early never removed an entry before the content replacing it was stored.
7. **Verify.** Plan every slice a last time; anything left to create, update or delete means the run did not converge and exits 1. The plan compares each stored entry's content hash and `meta_digest`, which the write itself commits, so a run with `--defer-generation` converges while the server is still generating.

Batches are packed by `--batch-size` and `--max-request-bytes` together and sent one after another, or up to `--concurrency` at once (see [Scheduling and Batch Sizing](#scheduling-and-batch-sizing)), and one failed document never rolls back the others. With `--dry-run` the run stops after the plan: it prints each create and update it would make and the deletes each slice's plan lists, at most 100 per slice followed by the number of further planned deletes, counts every planned delete, and writes nothing. Its counts and document outcomes are the planned ones, which equal what the apply run that follows reports when the corpus and the stored entries stay as they are and every write succeeds.

### Files and Directories the Run Cannot Use

- **A file that cannot be read**, is not valid UTF-8, holds nothing but whitespace, or whose create or update request alone exceeds `--max-request-bytes` is kept: the manifest carries a keep item for it, its stored entry, if any, stays as it is, the rest of the corpus is synchronized, and the run exits 1 naming the file.
- **A file whose `source_id` is longer than 256 characters** is left out of the manifest altogether, because no stored entry can carry such an identity, and the run exits 1 naming it. Nothing stored is deleted on its account.
- **A directory the walk cannot list** leaves every file below it out of the manifest, and the plan would call their stored entries absent. Such a run still creates and updates, but it executes no delete at all: it reports every planned delete as withheld and exits 1 naming the directory and the reason. The run after the permissions are fixed deletes whatever is really gone.

A walk that finds no document at all exits 2 unless the run passes `--allow-empty-corpus`, with which the run deletes every managed entry of the corpus. A mistyped path or an empty mount therefore never empties a corpus by accident.

### Exit Codes and the Report

| Code | Meaning                                                                                                                                                                                      |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0    | The thread matches the corpus; after `--dry-run`, every slice was planned and every file and directory was usable                                                                            |
| 1    | The run completed with problems: a document failed or was kept, a file was left out, a directory could not be listed, a request was too large, or the final plan still holds changes         |
| 2    | A configuration, authentication or plan error: an invalid flag, a refused token, a wrong `--server` path, a failed preflight, an empty corpus without `--allow-empty-corpus`, a refused plan |
| 69   | A transient failure: the server is unreachable, a call timed out or lost its connection or its response stream, or the server or a proxy answered 429, 502, 503 or 504; rerun later          |

When several apply, 2 outranks 69, which outranks 1. The command prints its progress and a summary on standard output and each run-level problem on standard error. The progress includes a line after every create and update batch that finishes, with the documents the leg applied so far, the batch's position and failed entries, how many of the leg's batches are done and the seconds since the leg began, such as `created 30 of 200 documents (batch 3 of 20, 1 failed, 3 of 20 batches done, 1872 s)`. With `--concurrency` above 1 the batches can finish out of order: each line names the batch that finished, and the documents applied and the batches done only ever grow. `--report PATH` writes the same outcome as JSON: `server`, `thread_id`, `dry_run`, `exit_code`, `counts` (`created`, `updated`, `deleted`, `unchanged`, `kept`, `foreign`, `failed`, `withheld` and `generation_pending`), `documents` (one `{source_id, outcome, context_id, error}` per document or removed entry) and `errors`. With `dry_run` true, the counts and the document outcomes are the planned ones, equal to what the apply run that follows reports when nothing changes in between and every write succeeds: `created`, `updated` and `deleted` count the writes that run would make, `deleted` counting every planned delete while `documents` and the printed lines name at most 100 deletes per slice, the ones each plan lists. A dry run whose plan succeeded with every file and directory usable ends its summary with `exit 0: planned; nothing was written`.

`generation_pending`, printed as `generation pending` in the summary, counts the created and updated entries the server answered with a `pending` [generation status](api-reference.md#generation-status); a dry run counts 0. A run without `--defer-generation` can count some too, such as a metadata update of an entry an earlier deferring run left pending. A pending entry is no problem of the run and changes no exit code: the server's generation worker stores the entry's embeddings and summaries after the run, and `get_context_by_ids` then reports the entry `ready`. An entry whose generation job fails every attempt turns `failed`: its document stays stored, readable and found by full-text search, and later runs report it unchanged, because its text and metadata match the file. Watch the `generation` counts of the server's `get_statistics`, and retry failed jobs by restarting the server or running `mcp-context-server-migrate --embed-missing` (see [Monitoring Deferred Generation](semantic-search.md#monitoring-deferred-generation)).

### Scheduling and Batch Sizing

Run the command on a schedule, from cron, a Kubernetes CronJob or a CI job, one run per corpus at a time (see [Concurrency and Recovery](#concurrency-and-recovery)). A run reads everything it needs from the server, so it keeps no local state and any machine can run it.

With generation on and without `--defer-generation`, the server generates every entry of a batch (embeddings, summary and per-section summaries) one after another before the batch commits, so the time of a batch call grows with its size. The default `--batch-size 10` keeps each call short enough to show progress and loses little when a call times out; raise it for a server with generation off, or for a run with `--defer-generation`, whose batch calls store the entries with their generation jobs and return without waiting for any generation. `--call-timeout` bounds one call and defaults to an hour; a proxy in front of the server needs timeouts at least as long (see [Limits](#limits)).

`--concurrency N` keeps up to N create and update batch calls in flight at once, each on a connection of its own; the default 1 sends them one after another. The batches start in order, a failed batch never stops the batches beside it, and the run records the outcomes in batch order, so a run that no call ends reports the counts, document outcomes and exit code of the same run sent one batch at a time. A call that ends the run, such as a timeout, keeps the batches not yet sent from starting and lets the ones in flight finish; the report then holds their outcomes too and names each distinct failure that ended the run. The deletes and every plan, the final one included, run one call at a time after the last create and update. Each document goes into exactly one batch, so the batches of one run never store a document twice. The run matches every per-entry result to its entry by index and assumes nothing about the order in which the server works through a batch.

The server bounds the generation work of all requests and of its generation worker together: `EMBEDDING_MAX_CONCURRENT` (3 by default) caps the embedding calls, `SUMMARY_MAX_CONCURRENT` (2 by default) the document and per-section summary calls, and `COMPRESSION_MAX_CONCURRENT` (the CPU count, at most 4, by default) the compression workers. A batch the semaphores cannot admit waits on the server, and the wait counts against `--call-timeout`. For a run without `--defer-generation`, keep the default `--concurrency 1` unless the summary model's server runs more than one request at once. The summary calls take most of a first ingest's time, and a model server that runs one request at a time, as Ollama does on the repository's test stacks with `OLLAMA_NUM_PARALLEL` unset, is already busy for nearly all of a run sent one batch at a time, so a higher value only queues batches behind it (see the measured costs below). When the model server runs several requests at once, set `--concurrency` to the server's `SUMMARY_MAX_CONCURRENT`, 2 by default: that setting caps how many summary calls run at once, so further batches mostly wait for the summary model. To go higher, raise `--concurrency`, `SUMMARY_MAX_CONCURRENT` and the model server's parallel requests (`OLLAMA_NUM_PARALLEL` for Ollama) together; on a CPU the parallel requests share the same cores, and those limits bound the gain of any value.

`--defer-generation` moves the generation out of the batch calls into the server's generation worker. The worker takes the queued entries oldest first, as many at once as `EMBEDDING_MAX_CONCURRENT` allows (`SUMMARY_MAX_CONCURRENT` on a server without an embedding provider), under the same limits as every request, so a higher `--concurrency` speeds up only the writes. The generation goes on after the run exits, and the server's other clients share the embedding and summary limits with the worker until it has drained the corpus's jobs. A deferring run therefore suits a corpus whose first ingest would otherwise outlast a proxy timeout or the schedule's window.

**Measured costs.** These figures come from the repository's test stacks (`deploy/docker/docker-compose.sqlite.ollama.test.yml` and `docker-compose.postgresql.ollama.test.yml`, the PostgreSQL stack with row-level security on, Ollama on CPU with the default models `qwen3-embedding:0.6b` and `qwen3:0.6b`). A client on the stack's Docker network timed the plan calls and the walk, and `mcp-context-ingest` on the Docker host timed the ingests. Treat the figures as orders of magnitude when you choose a schedule:

- A run that finds nothing to change spends about 1.1 to 1.4 s planning a corpus of 50,000 documents on either backend: 50 plan calls of 1,000 items, about 21 to 25 ms each. Row-level security adds under 1 ms per call.
- A `list_ingest_state` walk of 50,000 entries at page size 200 takes about 4 to 5 s.
- A first ingest of 200 documents of 1,500 to 4,000 characters, each with two to four sections, took about 8.7 s per document with embeddings, summaries and per-section summaries, sent one batch at a time (1,734 s on SQLite), with the summary model busy for 98 percent of that time. Against an Ollama that runs one request at a time, `--concurrency 2` took 9.2 s per document on SQLite (1,844 s) and 9.7 s on PostgreSQL (1,935 s); with `OLLAMA_NUM_PARALLEL=2` on the SQLite stack it took 7.3 s per document (1,455 s), about 16 percent less than the sequential run. A rerun over the same documents that found nothing to change took under 2 s.
- Building `idx_context_ingest` on an existing database of 50,000 managed entries takes well under a second on either backend.

### Running from the Docker Image

The server image installs the command on its `PATH`. To sync a directory of the host against a Compose stack, run it in a one-off container of the stack's server service, which reaches the server by its service name on the stack's network:

```bash
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml run --rm --no-deps \
  -v /path/to/handbook:/corpus:ro \
  --entrypoint mcp-context-ingest mcp-context-server \
  --server http://mcp-context-server:8000/mcp --corpus handbook --source-dir /corpus
```

Use the port the stack's server listens on (`FASTMCP_PORT`, 8000 in the shipped files). For a server that requires a token, export it in the shell and pass it through, adding `-e INGEST_TOKEN` before `--entrypoint` and `--token-env INGEST_TOKEN` after `--source-dir /corpus`; for a token a refresher rotates, use `--token-file` and mount the directory that holds the file, not the file itself, as the Docker guide shows. The image runs as UID 10001, so a file of the mounted tree that this user cannot read is kept and a directory it cannot list withholds the run's deletes; neither ever deletes an entry. The [Docker Deployment Guide](deployment/docker.md#running-the-ingest-cli-from-the-image) shows the same run for each stack.

## Concurrency and Recovery

**One sync per corpus at a time.** The server takes no lock for a sync, and with deduplication off nothing stops two overlapping runs from storing the same new document twice. Schedule runs of one corpus so that they never overlap, for example with `concurrencyPolicy: Forbid` on a Kubernetes CronJob or `flock` around a cron entry. Runs of different corpora may overlap freely.

**An interrupted run converges on the next one.** A run keeps no state of its own and derives every plan from what the server stores, so a run that crashed, lost its connection or was stopped leaves nothing to clean up by hand: the next run plans from the stored entries and finishes the work. A call that timed out on the client is never sent again within the run, because the server may still be working on it: a batch can commit after the client gave up, above all one that generates its entries before it commits, and a proxy timeout cuts the client off while the server goes on. A run with `--defer-generation` follows the same rule, and the next run's plan settles any such call. If such a batch, or an overlapping run, leaves two entries of one document, the next run's plan lists the extra one as a duplicate delete, and because the survivor order is total, every plan of that state names the same survivor.

**Edits by agents and other clients.** A sync restores the text of a managed entry that an agent or another client changed, because the stored content hash then differs from the file's. A change to a managed entry's metadata that keeps its block, through `update_context` with `metadata_patch` or a full `metadata` that carries the block, persists until the file's metadata or text changes, because the server compares `meta_digest` values and never recomputes them (see [The Ingest Metadata Block](#the-ingest-metadata-block)). A full `metadata` replacement without the block, or `metadata_patch` with `"ingest": null`, takes the entry out of the corpus: the sync stops managing it, never deletes it, and stores the document again as a new entry. A deleted managed entry is stored again by the next sync.

## Limits

| Limit                                  | Value                                    | Notes                                                                                                                          |
|----------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Manifest items per `plan_context_sync` | 1,000                                    | `mcp-context-ingest` plans a larger corpus in slices                                                                           |
| `source_id`                            | 256 characters                           | Enforced on every write, in the manifest and on the range bounds; the CLI leaves a file with a longer path out of the manifest |
| `thread_id`                            | 256 characters                           | So a corpus name is at most 253 characters                                                                                     |
| Deletes listed per plan                | 100                                      | `deletes_total` counts them all; delete the listed ones and plan again                                                         |
| `list_ingest_state` page               | 200 entries (default 100)                |                                                                                                                                |
| Entries per batch call                 | 100                                      | `--batch-size`, default 10                                                                                                     |
| HTTP request body                      | 4 MiB (4,194,304 bytes)                  | Fixed by the MCP SDK and not configurable; `--max-request-bytes` defaults to 3,145,728                                         |
| HTTP response read by a Python client  | 1 MiB (1,048,576 bytes) per SSE event    | Lifted by `FASTMCP_JSON_RESPONSE=true` on the server                                                                           |

The caps keep a maximal plan request under 4 MiB and a maximal response of either sync tool under 1 MiB, so the sync tools themselves never meet the transport limits.

**Large documents.** On the `http`, `streamable-http` and `sse` transports the MCP SDK refuses a request body above 4,194,304 bytes before any tool runs, and the limit cannot be raised, so a document above about 4.19 MB of UTF-8 text, less its metadata and the JSON around it, cannot be uploaded over HTTP. `mcp-context-ingest` keeps such a file instead of sending it, leaving its stored entry as it is, and exits 1 naming it; the stdio transport has no such limit.

**Large responses.** Python MCP clients cannot read a Server-Sent Events message above 1,048,576 bytes. The server answers every request of the initialize-handshake protocol era with Server-Sent Events, and a request of the sessionless era too when the call takes more than 15 seconds. A client that reads larger responses over HTTP, such as `get_context_by_ids` over several long documents, needs `FASTMCP_JSON_RESPONSE=true` on the server, which answers both eras with plain JSON.

**Reverse proxies.** A proxy in front of the server applies its own limits, and the common defaults sit below what a sync needs: ingress-nginx, which the [Helm](deployment/helm.md) and [Kubernetes](deployment/kubernetes.md) guides use, allows a 1 MiB request body and 60-second read and send timeouts, while `mcp-context-ingest` sends requests of up to 3 MiB by default and a batch with generation on can take many minutes unless the run passes `--defer-generation`. A proxy's 413 fails that call, with a message naming `--max-request-bytes`, `--batch-size` and the proxy body limit (exit 1 for a batch call, 2 for a plan call); a 504, or a response stream the proxy closes before the result arrives, ends the run with exit 69, and the server finishes the batch afterward, leaving at most a duplicate that the next run removes. Every batch keeps failing until the proxy is configured. Allow a body above the server's own 4 MiB limit, so that the server decides, and timeouts of at least `--call-timeout`.

For ingress-nginx, as Ingress annotations:

```yaml
nginx.ingress.kubernetes.io/proxy-body-size: "5m"
nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
```

For nginx, in the `server` or `location` block that proxies to the server:

```nginx
client_max_body_size 5m;
proxy_read_timeout 3600s;
proxy_send_timeout 3600s;
```

Behind a proxy you cannot reconfigure, lower `--max-request-bytes` under its body limit and `--batch-size` until a batch finishes within its timeout.

## Reading a Corpus

Agents read a corpus with the ordinary tools, scoped by `thread_id='kb:<corpus>'`:

- **Questions**: `hybrid_search_context(query=..., thread_id='kb:handbook')`, or `semantic_search_context` and `fts_search_context` with the same scope. Results carry a truncated text and the document's summary when the server generated one.
- **One document**: `search_context(thread_id='kb:handbook', metadata={'ingest.source_id': 'guides/setup.md'})` finds it by its source path, and `get_context_by_ids` returns its full text.
- **Long documents**: `navigate_context(context_id)` returns the document's outline of Markdown headings, with per-section summaries when called with `include_node_summaries=True`, and `read_context_range(context_id, node_id=...)` reads one section; see [Grep, Navigation & Partial Reads](grep-navigation-partial-read.md).
- **Exact text**: `grep_context(pattern, thread_id='kb:handbook')` finds literal or regular-expression matches with their positions.

Without `--defer-generation`, every create and every text update runs the server's generation steps before it commits, exactly as any other store: when generation is on and fails, the document is not stored or its old version stays, and the run records the failure and exits 1. Such a stored corpus entry carries every embedding and summary the server's settings call for. With `--defer-generation`, a document is readable and found by `search_context`, `fts_search_context`, `grep_context` and the full-text leg of `hybrid_search_context` as soon as the run stores it, while semantic search, its summary and its per-section summaries follow once the server's worker has generated them; every read reports the entry's `generation_status` (see [Generation Status](api-reference.md#generation-status)). A text update made this way removes the old text's embeddings and summaries when it stores the new text.

## Access Scope

Both sync tools resolve the caller once per call and see only what the caller may read. `plan_context_sync` manages only the caller's own entries: a readable managed entry of another principal only adds to `foreign_count`, and an entry the caller may not read is absent, neither counted nor planned, so a manifest item for its source plans a create of the caller's own copy. `list_ingest_state` lists only the caller's own entries. The plan is applied through the write tools, which apply their usual rules: a create belongs to its caller, and the plan names only the caller's own entries for updates and deletes. The ingest block never takes part in an access decision; see the [Access Model](authentication.md#access-model).

## Backend Parity

The sync tools behave identically on SQLite and PostgreSQL: the same entries are members of a corpus, the same manifest yields the same plan, and both list entries in the same order. Source ids compare by code point on both backends, through SQLite's default `BINARY` collation and PostgreSQL's `COLLATE "C"`, both of which order UTF-8 text by code point.

Both backends serve the sync reads from the partial index `idx_context_ingest`, which the server creates at startup and which holds only entries with an ingest block, so it stays empty until a corpus is stored. With row-level security on, the reads go through the read view and keep the index. On PostgreSQL the server also keeps extended statistics on the thread and the block's `scope` (`stat_context_ingest_scope`), created at startup like the index and refreshed by `ANALYZE` and autovacuum, so the planner knows that nearly every entry of a corpus thread is a member and walks the index in order instead of fetching and sorting the members. On SQLite the sync reads name the index (`INDEXED BY idx_context_ingest`), because SQLite chooses indexes from per-index averages that misjudge a corpus thread far larger than the database's average thread (see [SQLite](database-backends.md#sqlite-default)). PostgreSQL 14 or newer is required for these statistics.

## Configuration

`ENABLE_CONTEXT_SYNC` (`auto` by default) registers `plan_context_sync` and `list_ingest_state`. The tools need no extra dependency, so `auto` and `true` register them, and `false` leaves them out, in which case `mcp-context-ingest` stops at its preflight with exit 2 naming `plan_context_sync`. The caps of the [Limits](#limits) section are fixed protocol values, not settings. The settings that shape a corpus are those of the [Visibility](#visibility) section and, behind HTTP, `MCP_AUTH_PROVIDER` and `FASTMCP_JSON_RESPONSE`; see the [Environment Variables Reference](environment-variables.md).

## Related Documentation

- **API Reference**: [Sync Tools](api-reference.md#sync-tools) - `plan_context_sync` and `list_ingest_state`, the `deduplicate` parameter of [store_context](api-reference.md#store_context), and the [Generation Status](api-reference.md#generation-status) a deferring run leaves behind
- **Authentication**: [Authentication Guide](authentication.md) - tokens, JWT principals, the access model, sharing entries and anonymous read access
- **Metadata**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - the reserved `ingest` key and metadata filtering
- **Environment Variables**: [Environment Variables Reference](environment-variables.md) - `ENABLE_CONTEXT_SYNC` and the access-control settings
- **Database Backends**: [Database Backends Guide](database-backends.md) - SQLite, PostgreSQL and row-level security
- **Deployment**: [Docker](deployment/docker.md), [Helm](deployment/helm.md) and [Kubernetes](deployment/kubernetes.md) - running the CLI from the image and the reverse-proxy settings

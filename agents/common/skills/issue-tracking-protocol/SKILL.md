---
name: issue-tracking-protocol
description: |
  Cross-project task tracker in the issues thread of a context server. Use whenever you file an issue, bug report, feature request, improvement, or follow-up task for any project (your own or another); triage, prioritize, assign, or transition your own project's issue through its lifecycle; pick up an issue in your own project and implement it; comment on an issue; mark duplicates or blockers; or query the tracker (this project's open work and another project's rows when the user asks for them). Trigger on phrases like "file an issue", "create a task", "track this", "заведи задачу", "add it to the backlog", "start on issue N", on any call on the issues thread, on any request to record work for later or to act on work already recorded, or when agreed work in your own project will outlive the session -- even when no tracker is named. Not for an external tracker (YouTrack, Jira, GitHub, GitLab) that the user names or that the project's own convention uses for that work, nor the session's own to-do list.
---

<overview>

# Context-Server Issue Tracker

One unified tracker thread named `issues` holds every issue of every project -- one workspace-wide system of record, the Linear model. The project dimension lives in metadata (`project`), never in the thread name: any agent files an issue for any project into the same thread and filters by project when querying. One issue = one context entry; the entry's context ID is the issue number, globally unique across all projects by construction.

Why this shape: `project` metadata already encodes the project dimension, so a query is one `thread_id` plus metadata filters and agents need zero thread-name derivation; context IDs are global, so linking works identically within and across projects; the server's first-class `tags` are the single label mechanism; and one typed `links` object holds every connection an issue has, so no fact is ever stored twice.

**One thread is not one queue.** A session works from the queue of the project it is in. Rows belonging to other projects are visible here and are not available here: they are neither picked up nor put in front of the user as options, unless the user asked for that project in this session. Filing into another project's queue is a write the shared thread exists to allow; picking up what sits in that queue is not, and a filing is never a claim on the work it records.

</overview>

<schema_directive>

## Mandatory First Step: Load the Schema

Before composing or updating any tracker entry, invoke `Skill(skill="context-metadata-schema")`. That skill is the single normative source for the field contract this tracker relies on: the universal core (`schema_version`, `kind`, `project`, `links`), the issue and comment field tables, the status and priority vocabularies, the links registry with its three design rules, the filter recipes (including the compound kind+status MUST and the `case_sensitive` rule for string-ID filters), and the operational rules (read-modify-write link appends, supersession sequence, dedup caution, concurrency retry). The examples below show the contract in action but do not redefine it; on any discrepancy, the schema skill wins.

</schema_directive>

<before_filing>

## Before Filing: Find What Already Exists

Every new issue starts with a search for entries that may already record the same need. A second entry for one need splits its evidence and its discussion in two, and whoever picks up one of them never sees what the other holds. Search the target project's entries at every status, not only the open ones, because a closed entry on the same need is part of the answer:

- `hybrid_search_context(query="<the need in a few plain words>", thread_id="issues", metadata={"kind": "issue", "project": "<target>"}, limit=15)`, run with at least two phrasings -- the symptom as observed, and the component or capability it concerns -- since an earlier filer may have described the same need in other words.
- `search_context(thread_id="issues", metadata={"kind": "issue", "project": "<target>"}, limit=50)` to scan recent titles when the target project's queue is small enough to read whole.

Previews decide which entries are worth reading; read every plausible hit in full with `get_context_by_ids` before deciding, because a title can match while the body records a different need. What the search finds decides the write:

- **An open entry records the same need.** File nothing. Add what you know as a comment on it (`Re issue N: ...`) -- the new occurrence, evidence, or context -- and append the evidence to its `links.evidence` (read-modify-write). In your own project, also patch what the new information changes in its metadata, such as `priority`, and leave the body's observation as filed. In another project's entry the comment and the evidence link are the whole write: its priority, status, and body are that project's to change. Report that entry's number and your comment to whoever asked for the filing, in place of a new issue number.
- **An open entry overlaps but records a different need** (a narrower case, a follow-on, a sibling symptom): file the new entry with `links.related` naming the existing one, and say in its body how the two differ, so neither reader mistakes one for the other.
- **A `duplicate` entry matches.** Its wording is often what the search finds, while the need lives on in the entry its `links.duplicate_of` names, possibly in another project. Follow that link -- again if the entry it names is a duplicate too -- and apply these rules to the entry at the end of it.
- **A closed entry (`done` or `canceled`) records the same need.** The closed record is history and stays as it is. When the need is back -- a regression after a fix, new evidence that answers a cancellation -- file a new entry with `links.related` naming it and a body saying what changed since it closed.
- **Nothing records the need.** File it as the next section describes.

**Two open entries for one need are merged, not left side by side** -- whether this search or a triage pass finds them, in your own project's queue. Keep as canonical the entry the work has advanced furthest (an assignee, progress, discussion), else the older one. Carry into it what only the other holds: its evidence and acceptance criteria as a comment on the canonical entry, its tags, and its `links` edges merged into the canonical entry's arrays (read-modify-write); re-point any sub-issue whose `parent`, and any issue whose `blocks`, names the duplicate. Then close the other with `{"status": "duplicate", "links": {"duplicate_of": [<canonical-id>]}}` and a comment naming the canonical entry, so the need survives in one place and nothing still points at the closed one. Duplicates sitting in another project's queue are that project's to merge: comment on the entry your need matches and leave the rest to its owner.

</before_filing>

<own_project_work>

## Agreed Work That Outlives the Session

Work the user agreed to in this project -- a request, a planned change, a bug they reported or the session found -- lives in the session's plan while the session carries it out. It belongs in this project's own queue as soon as it will outlive that session: the user deferred it ("later", "after X", "add it to the backlog"), it waits on something outside the session (a change in another project, a release, a step only the user can take), it falls outside the plan being executed, or the session is about to end with it unfinished. File it then, after the search above, at `todo` or `backlog`, with `links.commissioned_by` naming the user message that agreed it, and name the entry in the plan that carried it. Work that starts and finishes inside the session that agreed it is not filed: the plan already records it, and an entry opened only to be closed minutes later doubles the record and hides the queue's real backlog.

</own_project_work>

<filing>

## Filing an Issue

Store a new entry in the `issues` thread only once the search above found nothing that records the need. The body is Markdown: H1 title first line, then Problem/Goal, Context/Evidence, and Acceptance criteria sections, substance front-loaded because search previews truncate from the start. Give every body concrete specifics early -- identical-text deduplication collapses template-only bodies filed twice.

```text
store_context(
    thread_id="issues",
    source="agent",
    text="# Fix flaky teardown in integration tests\n\n## Problem\n...\n\n## Context\n...\n\n## Acceptance criteria\n...",
    metadata={
        "schema_version": 1,
        "kind": "issue",
        "project": "target-project",
        "source_project": "project-where-discovered",
        "agent_name": "main-agent",
        "title": "Fix flaky teardown in integration tests",
        "status": "triage",
        "priority": 3,
        "links": {"commissioned_by": [27318], "evidence": [27326]}
    },
    tags=["issue", "bug", "tests"]
)
```

Rules that matter at filing time: an entry is filed only on real-world evidence of the need it records, cited under Context/Evidence -- for a defect, an observed request, incident, log, record, or user report showing the failure occurs, or, for code not yet in use, showing that real use produces its triggering input; for a feature, improvement, hardening, or debt entry, a user's request or a real case the current behavior fails to serve; for a case that a rule or skill in force declares a defect in its own right, that case itself, located in the artifact -- because an entry built on a crafted input, a placeholder value, a test-manufactured condition, or a hypothetical such as "it could happen" or "defense in depth" records work nobody needs and then sits in a queue asking to be done; when the evidence check finds no real occurrence, record that check where the finding arose and file nothing. `status` defaults to `triage` when you file into a project you do not own (the owning project's sessions accept it into `backlog`/`todo`); a project owner may file at any status. `priority` is numeric with `0` meaning none, not top. `tags` are `["issue"]` plus labels -- recommended core label vocabulary, lowercase kebab-case: `bug`, `feature`, `improvement`, `tech-debt`, `docs`, `security`, `dx`, `question`; free extension is fine. Report the returned context ID as the issue number.

An entry recording a change your own project needs from the target project carries `links.blocks` naming your project's entry for the work that waits on it, filed in your own queue first when none exists yet, so your later sessions find what they wait on with the recipe below and take the change up when it lands.

</filing>

<lifecycle>

## Updating an Issue

Patch only what changes, via RFC 7396 `metadata_patch`:

```text
update_context(context_id=27401, metadata_patch={"status": "in_progress", "assignee": "main-agent"})
update_context(context_id=27401, metadata_patch={"status": "done", "completed_at": "2026-08-01T15:00:00Z"})
```

Set `completed_at` whenever status becomes `done` or `canceled`. Closing as duplicate is a pair: `metadata_patch={"status": "duplicate", "links": {"duplicate_of": [<canonical-id>]}}`. Adding any link edge is read-modify-write: read the issue's current `links`, append client-side, patch the touched key with its complete new array (sibling link keys survive the deep merge). Label changes go through the `tags` parameter, which replaces the whole list -- resend every tag including `issue`. Text edits replace the whole body. On a version-conflict error, refetch and reapply once.

</lifecycle>

<commenting>

## Commenting on an Issue

A comment is a new entry in the same thread -- never append discussion into the issue's own text:

```text
store_context(
    thread_id="issues",
    source="agent",
    text="Re issue 27401: reproduced on Windows only; the teardown races the file-lock release. Evidence in report 27455.",
    metadata={
        "schema_version": 1,
        "kind": "comment",
        "project": "target-project",
        "agent_name": "main-agent",
        "links": {"parent": [27401], "evidence": [27455]}
    },
    tags=["comment"]
)
```

Comments attach to exactly one parent, never to another comment (discussions are flat in schema v1), and always open the body by naming the parent (`Re issue 27401:`) -- that keeps them readable standalone and prevents short identical bodies from dedup-collapsing across different parents. Reading a discussion: filter kind `comment` plus `array_contains` on `links.parent` with the issue ID. Read it before commenting, so a comment adds what the discussion does not already say.

</commenting>

<queries>

## Query Recipes

All queries run against `thread_id="issues"` unless noted; search results are truncated previews for relevance triage -- retrieve full bodies with `get_context_by_ids` before acting on substance.

**`<name>` below is the canonical project name of the session you are in, and every query that sources work uses it; it changes only when the user asked for another project in this session.** The recipes that omit the project key answer a narrow question and are not a way to find something to do; what they return for other projects is context rather than a candidate to work on or to offer, unless the user asked for that project in this session.

- **A project's open work:** open means the five non-terminal statuses, so ask for all of them at once -- `search_context(thread_id="issues", metadata={"kind": "issue", "project": "<name>"}, metadata_filters=[{"key": "status", "operator": "in", "value": ["triage", "backlog", "todo", "in_progress", "in_review"]}])`. Narrowing to one status (`metadata={"kind": "issue", "project": "<name>", "status": "todo"}`) answers a different question -- what is already accepted and scheduled -- and is right only when that is the question you have.
- **Cross-project urgency, on request only:** kind `issue` plus `priority gt 0 AND priority lt 3`, no project filter. Run it when the user asks what is urgent across projects, and answer the question they asked.
- **By label:** the `tags` parameter (OR semantics); for a rare label-AND, run the narrower tag query and intersect client-side.
- **By assignee:** `metadata={"kind": "issue", "project": "<name>", "assignee": "<who>"}`.
- **Provenance:** `metadata={"kind": "issue", "source_project": "<name>"}` finds issues discovered while working in a project, wherever they target. It answers where a need was found; the rows targeting other projects remain those projects' work.
- **Sub-issues of X vs discussion of X:** same `array_contains` on `links.parent` with value X, disambiguated by `kind` (`issue` = sub-issues, `comment` = discussion). Never run a `links.parent` query without a `kind` filter.
- **What blocks X:** `array_contains` on `links.blocks` with value X (plus kind `issue`); what X blocks is X's own `links.blocks` array.
- **What your project waits on in other projects:** for each entry of your project's open work, `search_context(thread_id="issues", metadata={"kind": "issue"}, metadata_filters=[{"key": "links.blocks", "operator": "array_contains", "value": <entry-id>}])`, keeping the rows whose `project` is not `<name>`. An open row is a change still pending there; a `done` one means the change landed, so take it up in the waiting entry and retire any interim measure that stood in for it.
- **Duplicates of X:** `array_contains` on `links.duplicate_of` with value X.
- On a server issuing string (hex) context IDs, every links filter above adds `"case_sensitive": true`; integer IDs need no flag.

</queries>

<triage>

## Triage Discipline

**The trigger is a query, not a schedule.** Whenever you query your own project's open work and the result holds entries with status `triage`, those entries are the queue, and dispositioning them is part of that query -- do it before choosing what to work on. An entry sitting in `triage` is a request nobody has answered yet, and a queue you have not dispositioned cannot tell you what is worth doing next. A session that never consults the tracker owes nothing here; the obligation attaches to the moment you look.

The queue you triage is your own project's, and so is the queue you work from. An entry for any other project -- at any status, whether you filed it there or passed it while querying -- belongs to that project's queue, its sessions meet the same trigger there, and reporting its existence from here changes nothing, so leave it alone and say nothing about it, except to report a filing this session itself made, or the comment it added in place of one, which is a completed action rather than an offer, or to name an entry recording a change your own project's work waits on, which reports your project's need. That covers offering it as work too: another project's issue is never a candidate to propose here unless the user asked for that project in this session.

**Which transitions are yours to make.** The line runs between a disposition you can justify with evidence and one that rests on your preference. Accepting needs no justification, because the filer already made the case: move the entry to `todo` (work the project intends to reach) or `backlog` (accepted, not scheduled) on your own initiative. `duplicate` is a factual finding -- when an existing issue demonstrably covers the same need, the need survives there rather than being discarded. When that issue is open, merge the two as Before Filing describes. When it is closed and the entry in `triage` adds nothing new, close the entry with `{"status": "duplicate", "links": {"duplicate_of": [<closed-id>]}}` and a comment naming the closed issue and why it covers this one. When the entry shows the need is back -- a regression, new evidence the closure did not weigh -- it is not a duplicate: accept it and add `links.related` naming the closed issue. `canceled` splits in two: cancel on your own initiative only when the reason is verifiable and you verified it (the failure no longer occurs in real use, the artifact no longer carries the case a rule or skill in force declares a defect, the capability has since shipped, or the entry rests only on a synthetic reproduction or an imagined case and an evidence check of real requests, incidents, logs, records, and user reports finds no occurrence), stating that check and its result in the closing comment -- a synthetic-only entry, whatever its label, is canceled this way at any status, never kept in a backlog; when the reason is instead that the work looks not worth doing, the decision is the user's, because the session that filed it saw something you cannot see from here.

**When the decision is not yours,** leave the entry in `triage` and put the proposal to the user through the structured question tool, naming each entry and what you would do with it, batched into one question rather than a stream of them. An entry still in `triage` because nobody has answered is in the right state -- unaccepted, visible to the next query, not lost -- so never cancel by default to empty a queue.

**What triage produces is the transitions.** Report what you changed. Do not additionally warn that unaccepted entries exist, or that a narrower query would have hidden them: you ran the query that does not hide them, and the entries it surfaced are now handled. A warning repeated every session in place of a disposition is the failure this section exists to prevent.

</triage>

<picking_up>

## Picking Up an Issue

**Take up your own project's work only.** The issue you start belongs to the project this session is in; another project's issue is not a candidate here, however well it fits or however cheap it looks, unless the user asked for that project in this session. When your own open work comes to one entry or none, that is the answer -- report it and stop, rather than widening the query until a longer list appears or asking the user whether to widen it. When your work needs a change another project should make, that change is that project's to make: it goes into that project's queue, as an entry or a comment on one already recording it, and is never implemented from here without the user's explicit permission, while your entry for the waiting work stays in your queue until the change lands.

**Re-verify what the work will rest on before implementing it.** The first claim to check is the evidence itself: what real request, incident, log, record, or user report shows the need occurring, and whether it still does -- or, for an entry recording a case that a rule or skill in force declares a defect in its own right, whether the artifact still carries that case. An entry that proves to rest only on a synthetic reproduction or an imagined case, whenever and by whomever it was filed, has nothing to implement. An issue body is a point-in-time observation written in the present tense, and that tense is exactly what makes a stale reading look like current state: an entry saying that a queue is empty, that an endpoint answers in some shape, or that a file still carries a line reads identically on the day it was filed and a month after it stopped being true. So take the live claims the implementation will stand on -- states read from an API, a queue, or the filesystem, timestamps, and anything else phrased in the present tense -- and check them against the artifact itself before building anything on them. Proportionate means exactly that set: the claims the work rests on, never every sentence in the body, and never the background it merely passes.

**Acceptance criteria inherit the errors of the evidence they were derived from.** Criteria are written out of those same observations, so a stale or misread one yields a criterion that still looks satisfiable while prescribing the wrong thing -- and an agent that meets it faithfully ships something the artifact itself refutes, with every check green. A criterion whose ground has moved is therefore not satisfied by following it literally. Implement what is true and let the criteria follow the evidence, never the reverse.

**Record the divergence instead of quietly absorbing it.** A comment carries it -- when found, or in the closing one at the latest -- naming what the body claimed, what the artifact actually shows, how that was established, and what was built instead. Leave the body's observation as filed: it records what was seen at that moment, and rewriting it to match the newer reading destroys the only trace that the reading ever changed.

**When re-verification leaves nothing to implement, the entry is closed rather than built.** A failure that no longer occurs in real use, a declared case the artifact no longer carries, a capability that has since shipped, and an entry whose only basis proves synthetic all clear the evidence bar the triage section above sets for canceling an entry on your own initiative -- verifiable, and verified by you, with that check stated in the closing comment. When instead the premise is gone but real-world evidence shows the need behind it survives, the call is the user's: leave the entry open and put it to them through the structured question tool.

</picking_up>

<anti_patterns>

## Anti-Patterns (Forbidden)

- Never store work reports, session context, or knowledge-base notes in the `issues` thread: it holds only kind `issue` entries and their kind `comment` discussion.
- Never file an issue before searching for an entry that already records the need, and never leave two open entries for one need in your own project's queue.
- Never create per-project tracker threads -- the project dimension is metadata.
- Never duplicate labels into a metadata field -- tags are the single label mechanism.
- Never store both directions of an edge, and never split connections across a second field -- one typed `links` object holds every connection, and the reverse direction is a query.
- Never mix status vocabularies -- issue statuses never appear on reports, and `pending` never appears on issues.
- Never comment on a comment, and never append discussion into an issue's body.
- Never renumber or reuse issue IDs -- context IDs are immutable.
- Never present another project's issues as work available here unless the user asked for that project in this session, and never widen a query past your own project to produce a longer list of options; naming an entry that records a change your own project waits on reports your project's need and is no such offer.
- Never file secrets, tokens, or credentials into issue bodies or comments.

</anti_patterns>

<walkthrough>

## Lifecycle Walkthrough

1. File: the target project's entries, searched at every status, hold nothing on the need, so `store_context` into `issues` with kind `issue`, status `triage` (foreign project), priority 3, `links.commissioned_by` pointing at the user message -- returns ID 27401.
2. Accept: a session working in the owning project queries its own open work, finds 27401 sitting in `triage`, and patches `{"status": "todo"}`.
3. Start: re-verify the live claims the work will rest on, then patch `{"status": "in_progress", "assignee": "main-agent"}`.
4. Discuss: a kind `comment` entry with `links.parent: [27401]`, body opening `Re issue 27401:`.
5. Link: discovery that 27401 blocks 27130 -- read 27401's `links`, patch `{"links": {"blocks": [27130]}}`.
6. Close: `{"status": "done", "completed_at": "..."}`, plus a closing comment citing the shipping commit in `links.git_commits` and stating any divergence from the body's evidence.

</walkthrough>

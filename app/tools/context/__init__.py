"""
Context CRUD operations for MCP tools.

This package contains the core context management tools:
- store.py: store_context, store a new context entry
- retrieve.py: get_context_by_ids, retrieve specific context entries by ID
- update.py: update_context, update an existing context entry
- delete.py: delete_context, delete context entries by ID or by thread

Generation-First Transactional Integrity (the default):
The store and update tools implement atomic generation + data storage. When embedding or summary
generation is enabled, all generation runs OUTSIDE any database transaction via
run_generation() (app.tools._generation), which drives three concurrent legs:
1. embed_then_compress -- embedding followed by TurboQuant compression (abort-mandatory)
2. the flat document summary (abort-mandatory)
3. the index_tree per-node summaries (never-raise), started after the flat summary
   completes and overlapped with the embedding leg
If either abort-mandatory leg fails after exhausting its retries (managed by
app/embeddings/retry.py and app/summary/retry.py), the failures are collected into a
single deterministic ToolError and NO data is saved; the never-raise node leg is then
cancelled and awaited before the transaction opens. A node-summary failure or timeout
never aborts the store. Only when both abort-mandatory legs succeed do ALL database
operations occur in a SINGLE atomic transaction.

Deferred generation:
With defer_generation=True, while an embedding or a summary provider is active, a store or a text
update generates nothing before its transaction: the transaction commits the entry together with
a generation job, and the generation worker (app.generation_jobs) generates and stores the outputs
after the commit. The response's generation_status is 'pending' until the job completes ('ready')
or spends every attempt ('failed').

Infrastructure shared with the batch tools in app/tools/batch/ lives in single-purpose modules: embedding/summary generation in
app.tools._generation, the store and update transactions in app.tools._store_transaction and
app.tools._update_transaction, the heartbeat, connection error classification and control-flow
exceptions they share in app.tools._transactions, input and image validation in app.tools._validation,
delete-path embedding cleanup in app.tools._delete_cleanup, and response message builders in
app.tools._responses.
"""

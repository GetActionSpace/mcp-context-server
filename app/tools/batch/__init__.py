"""
Batch operations for MCP tools.

This package contains tools for bulk context management:
- store.py: store_context_batch, store multiple entries in one operation
- update.py: update_context_batch, update multiple entries in one operation
- delete.py: delete_context_batch, delete entries by various criteria
- entry_validation.py: per-entry validation shared by the store and update tools
- update_access.py: the update tool's write-access checks (pre-generation authorization of every
  update, and the in-transaction re-probe that disambiguates a compare-and-set matching zero rows)

Generation-First Transactional Integrity:
The store and update tools implement atomic generation + data storage for batch operations.
Each entry's embedding + summary are generated in PARALLEL via asyncio.gather(return_exceptions=True),
reusing generate_embeddings_with_timeout and generate_summary_with_timeout from app.tools._generation.
Entries within a batch are processed SEQUENTIALLY. When generation is enabled:
1. Per-entry: embeddings and summaries run in parallel OUTSIDE any database transaction
2. If ANY generation fails in atomic mode, NO data is saved
3. If all generation succeeds, ALL database operations occur in a SINGLE atomic transaction

Infrastructure shared with the context tools in app/tools/context/ lives in single-purpose modules:
embedding/summary generation in app.tools._generation, the store and update transactions in
app.tools._store_transaction and app.tools._update_transaction, the heartbeat, connection error
classification and control-flow exceptions they share in app.tools._transactions, input and image
validation in app.tools._validation, delete-path embedding cleanup in app.tools._delete_cleanup, and
response message builders in app.tools._responses.
"""

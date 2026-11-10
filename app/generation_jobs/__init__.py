"""Deferred generation: the worker that drains the generation job queue and the executor it runs.

A write that defers generation stores its entry together with a generation job. The
worker (``worker.py``) claims the jobs oldest first and runs each through the executor
(``executor.py``), which generates the outputs the entry still lacks and stores them in one
transaction with the job's removal. Neither module holds SQL or chooses an identity: the
repositories run the statements, and the identity the jobs run as is handed in by whoever
starts them.
"""

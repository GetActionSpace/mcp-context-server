"""
Base protocol definition for storage backends.

This module defines the StorageBackend Protocol that all backend implementations
must follow. The protocol ensures type-safe, database-agnostic interfaces for
repositories and server components.

The protocol uses @runtime_checkable to enable isinstance() checks and TypeVar
for generic operation signatures that preserve return types.

Every entry point takes the identity it runs as through a required keyword-only
``access_scope``: a request scope, the system scope, or :data:`MAINTENANCE` for the
migrations, startup and command-line paths that run outside any request. A backend
that enforces access in the database binds that identity before the operation runs;
one that enforces access only in the application layer accepts and discards it.

A plain read names each table it selects from through ``readable_relation``, so a
backend can substitute a relation that applies its access rules to the same rows.
"""

import re
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import final
from typing import overload
from typing import runtime_checkable

from app.access_scope import Scope

T = TypeVar('T')


@final
class MaintenanceAccess:
    """The identity of schema maintenance, startup and command-line work outside any request.

    It carries no principal and is never built from request data. Only migrations,
    startup, the command-line tools and the compression provenance run as it; the
    tools and repositories always run as the caller.
    """

    __slots__ = ()


MAINTENANCE = MaintenanceAccess()

type BackendAccess = Scope | MaintenanceAccess

# The tables a plain read names through ``readable_relation``: the context table and every
# child table that references it. The grants table is absent because only the access
# predicate reads it, by its own name.
ROUTABLE_TABLES = frozenset({
    'context_entries',
    'tags',
    'image_attachments',
    'embedding_metadata',
    'embedding_chunks',
    'vec_context_embeddings',
    'vec_context_embeddings_compressed',
    'context_index_nodes',
    'generation_jobs',
})

_RELATION_ALIAS = re.compile(r'[a-z_][a-z0-9_]*')


def base_relation(table: str, *, alias: str) -> str:
    """Render the relation a plain read names ``table`` by when no access view replaces it.

    Args:
        table: A routable table.
        alias: The name the statement qualifies the table's columns with; the table's own
            name when the statement uses no alias.

    Returns:
        ``table`` when ``alias`` is the table's own name, otherwise ``'<table> <alias>'``.

    Raises:
        ValueError: If ``table`` is not routable or ``alias`` is not a lowercase identifier.
    """
    if table not in ROUTABLE_TABLES:
        raise ValueError(f'{table!r} is not a routable table')
    if _RELATION_ALIAS.fullmatch(alias) is None:
        raise ValueError(f'{alias!r} is not a valid relation alias')
    return table if alias == table else f'{table} {alias}'


@runtime_checkable
class TransactionContext(Protocol):
    """Protocol for transaction context providing connection access.

    TransactionContext is yielded by begin_transaction() and provides:
    - Access to the underlying database connection
    - Backend type identification for SQL generation

    The transaction lifecycle (begin, commit, rollback) is managed by
    the begin_transaction() context manager, not by TransactionContext itself.

    Example Usage:
        async with backend.begin_transaction(access_scope=scope) as txn:
            # For PostgreSQL: txn.connection is asyncpg.Connection
            # For SQLite: txn.connection is sqlite3.Connection

            if txn.backend_type == 'postgresql':
                await txn.connection.execute('INSERT INTO ...')
            else:
                txn.connection.execute('INSERT INTO ...')
    """

    @property
    def connection(self) -> object:
        """Get the underlying database connection.

        Returns:
            For SQLite: sqlite3.Connection
            For PostgreSQL: asyncpg.Connection
        """
        ...

    @property
    def backend_type(self) -> str:
        """Get the backend type identifier.

        Returns:
            'sqlite' or 'postgresql'
        """
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol defining the interface for storage backend implementations.

    All storage backends (SQLite, PostgreSQL, etc.) must implement
    this protocol to ensure compatibility with the repository layer.

    The protocol defines methods for:
    - Lifecycle management (initialize, shutdown)
    - Connection management (get_connection)
    - Operation execution (execute_write, execute_read)
    - Read routing (readable_relation)
    - Health monitoring (get_metrics, backend_type)

    Example Implementation:
        class MySQLBackend:
            async def initialize(self) -> None:
                # Connect to MySQL, set up connection pool
                pass

            async def shutdown(self) -> None:
                # Close all connections gracefully
                pass

            @asynccontextmanager
            async def get_connection(self, readonly: bool = False) -> AsyncGenerator[Any, None]:
                # Yield a MySQL connection from pool
                conn = await self.pool.acquire()
                try:
                    yield conn
                finally:
                    await self.pool.release(conn)

            async def execute_write(
                self, operation: Callable[..., T], *args, access_scope: BackendAccess, **kwargs,
            ) -> T:
                # Execute write operation with retry logic as the given identity
                async with self.get_connection(readonly=False) as conn:
                    return operation(conn, *args, **kwargs)

            async def execute_read(
                self, operation: Callable[..., T], *args, access_scope: BackendAccess, **kwargs,
            ) -> T:
                # Execute read operation as the given identity
                async with self.get_connection(readonly=True) as conn:
                    return operation(conn, *args, **kwargs)

            def readable_relation(self, table: str, *, alias: str) -> str:
                return base_relation(table, alias=alias)

            def get_metrics(self) -> dict[str, Any]:
                return {
                    'backend_type': 'mysql',
                    'pool_size': self.pool.get_size(),
                    'active_connections': self.pool.get_idle_size(),
                }

            @property
            def backend_type(self) -> str:
                return 'mysql'

    Example Usage:
        # Create backend
        backend = SQLiteBackend(db_path='/path/to/db.sqlite')
        await backend.initialize()

        # Use with repository
        repo = ContextRepository(backend)

        # Execute operations
        async with backend.get_connection(readonly=True) as conn:
            results = conn.execute('SELECT * FROM context_entries').fetchall()

        # Get health metrics
        metrics = backend.get_metrics()
        print(f"Backend: {backend.backend_type}, Metrics: {metrics}")

        # Cleanup
        await backend.shutdown()
    """

    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        This method is called once during server startup to establish connections,
        create connection pools, and perform any necessary setup.

        For SQLite:
            - Creates database file if not exists
            - Initializes connection pool (readers + writer)
            - Starts background health check task
            - Loads sqlite-vec extension

        For PostgreSQL:
            - Creates asyncpg connection pool
            - Detects Pgpool-II for connection pooler compatibility
            - Configures statement cache

        Raises:
            RuntimeError: If initialization fails
            ConnectionError: If database is unreachable

        Example:
            backend = SQLiteBackend(db_path='/data/context.db')
            await backend.initialize()
            # Backend is now ready for use
        """
        ...

    async def shutdown(self) -> None:
        """
        Gracefully shut down the storage backend.

        This method is called during server shutdown to close all connections,
        cancel background tasks, and release resources.

        For SQLite:
            - Cancels health check task
            - Closes all reader connections
            - Closes writer connection
            - Ensures no connections are leaked

        For PostgreSQL:
            - Closes asyncpg pool
            - Waits for in-flight queries to complete
            - Releases all connections

        Raises:
            RuntimeError: If shutdown is called before initialization

        Example:
            await backend.shutdown()
            # All connections closed, resources released
        """
        ...

    def get_connection(
        self,
        readonly: bool = False,
        allow_write: bool = False,
    ) -> AbstractAsyncContextManager[Any]:
        """
        Get a database connection from the pool.

        This is an async context manager that yields a connection and ensures
        proper cleanup on exit.

        Args:
            readonly: If True, returns a read-only connection (for SQLite, a reader)
            allow_write: If True, allows write operations on readonly connection
                        (for SQLite, this is used by _execute_write_internal)

        Yields:
            Connection object (sqlite3.Connection for SQLite, asyncpg.Connection for PostgreSQL)

        For SQLite:
            - readonly=True: Returns connection from reader pool
            - readonly=False: Returns the single writer connection
            - Connections are automatically returned to pool on exit

        For PostgreSQL:
            - Both readonly and write connections come from the same pool
            - readonly parameter is advisory (PostgreSQL handles this via transactions)

        Raises:
            RuntimeError: If backend is shut down or not initialized

        Example:
            async with backend.get_connection(readonly=True) as conn:
                cursor = conn.execute('SELECT * FROM context_entries')
                results = cursor.fetchall()
            # Connection automatically returned to pool
        """
        ...

    @overload
    async def execute_write(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_write(
        self,
        operation: Callable[..., T],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T: ...

    async def execute_write(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T:
        """
        Execute a write operation with retry logic and proper connection handling.

        This method wraps write operations with:
        - Retry logic with exponential backoff
        - Circuit breaker protection
        - Connection management
        - Error handling

        Args:
            operation: Sync or async callable that performs the write operation.
                      SQLite backend expects sync callable: Callable[..., T]
                      PostgreSQL backend expects async callable: Callable[..., Awaitable[T]]
            *args: Positional arguments to pass to operation
            access_scope: The identity the write runs as. The backend consumes it;
                it never reaches ``operation``.
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        For SQLite:
            - Queues write operation in write queue (serialized execution)
            - Uses single writer connection
            - Retries on transient errors (SQLITE_BUSY, SQLITE_LOCKED)
            - Circuit breaker prevents cascading failures
            - Discards access_scope: access is enforced in the application layer
            - Expects sync callable: operation(conn, *args, **kwargs) -> T

        For PostgreSQL:
            - Uses connection from pool
            - Wraps in transaction
            - Binds access_scope through the session profile as the first statement
              of every attempt's transaction
            - Retries on connection errors
            - Expects async callable: await operation(conn, *args, **kwargs) -> T

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open
            Exception: Original exception from operation after retries exhausted

        Example (SQLite - sync):
            from app.ids import generate_id

            def insert_context(conn, text, thread_id):
                new_id = generate_id()
                conn.execute(
                    'INSERT INTO context_entries (id, text_content, thread_id) VALUES (?, ?, ?)',
                    (new_id, text, thread_id),
                )
                return new_id

            context_id = await backend.execute_write(insert_context, 'Hello', 'thread-123', access_scope=scope)

        Example (PostgreSQL - async):
            from app.ids import generate_id

            async def insert_context(conn, text, thread_id):
                new_id = generate_id()
                await conn.execute(
                    'INSERT INTO context_entries (id, text_content, thread_id) VALUES ($1, $2, $3)',
                    new_id, text, thread_id,
                )
                return new_id

            context_id = await backend.execute_write(insert_context, 'Hello', 'thread-123', access_scope=scope)
        """
        ...

    @overload
    async def execute_read(
        self,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def execute_read(
        self,
        operation: Callable[..., T],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T: ...

    async def execute_read(
        self,
        operation: Callable[..., T] | Callable[..., Awaitable[T]],
        *args: Any,
        access_scope: BackendAccess,
        **kwargs: Any,
    ) -> T:
        """
        Execute a read operation with proper connection handling.

        This method wraps read operations with:
        - Read-only connection from pool
        - Error handling
        - Connection cleanup

        Args:
            operation: Sync or async callable that performs the read operation.
                      SQLite backend expects sync callable: Callable[..., T]
                      PostgreSQL backend expects async callable: Callable[..., Awaitable[T]]
            *args: Positional arguments to pass to operation
            access_scope: The identity the read runs as. The backend consumes it;
                it never reaches ``operation``.
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation (type preserved via TypeVar)

        For SQLite:
            - Uses connection from reader pool
            - No retry logic needed (reads don't cause locks)
            - Load balancing across multiple readers
            - Discards access_scope: access is enforced in the application layer
            - Expects sync callable: operation(conn, *args, **kwargs) -> T

        For PostgreSQL:
            - Uses connection from pool
            - When the session profile wraps reads, runs the operation inside a
              transaction whose first statement binds access_scope; otherwise runs
              it as issued
            - Expects async callable: await operation(conn, *args, **kwargs) -> T

        Example (SQLite - sync):
            def get_context_by_id(conn, context_id):
                cursor = conn.execute(
                    'SELECT * FROM context_entries WHERE id = ?',
                    (context_id,)
                )
                return cursor.fetchone()

            entry = await backend.execute_read(get_context_by_id, 123, access_scope=scope)

        Example (PostgreSQL - async):
            async def get_context_by_id(conn, context_id):
                row = await conn.fetchrow(
                    'SELECT * FROM context_entries WHERE id = $1',
                    context_id
                )
                return row

            entry = await backend.execute_read(get_context_by_id, 123, access_scope=scope)
        """
        ...

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name the relation a plain read selects ``table``'s rows from.

        A plain read is a SELECT that writes nothing and takes no row lock. It names each
        routable table it reads through this method, and qualifies the table's columns
        and its access predicate with ``alias``. Name resolution is the same in every
        form: the base form keeps the alias (``context_entries ce``), and a backend that
        reads through an access view names the view under the same alias
        (``<view> AS ce``).

        Args:
            table: A routable table (:data:`ROUTABLE_TABLES`).
            alias: The name the statement qualifies the table's columns with; the table's
                own name when the statement uses no alias.

        Returns:
            The FROM or JOIN item to place in the statement.

        Raises:
            ValueError: If ``table`` is not routable or ``alias`` is not a lowercase
                identifier.

        For SQLite:
            - Always the base form (:func:`base_relation`): access is enforced in the
              application layer

        For PostgreSQL:
            - Delegates to the session profile, which maps a table to its access view
              or renders the base form

        Example:
            relation = backend.readable_relation('context_entries', alias='ce')
            sql = f'SELECT ce.id FROM {relation} WHERE ce.thread_id = ?'
        """
        ...

    def begin_transaction(self, *, access_scope: BackendAccess) -> AbstractAsyncContextManager['TransactionContext']:
        """Begin a database transaction that can span multiple operations.

        This method returns an async context manager that yields a TransactionContext.
        All operations executed within the context share the same transaction:
        - On successful exit: transaction is committed
        - On exception: transaction is rolled back

        Use this when multiple repository operations must succeed or fail atomically.

        For single operations, continue using execute_write() which handles
        transactions automatically.

        Args:
            access_scope: The identity every operation in the transaction runs as.

        Yields:
            TransactionContext with access to the connection and backend_type

        For SQLite:
            - Acquires writer connection with exclusive lock
            - Begins deferred transaction implicitly
            - Commits on context exit, rollbacks on exception
            - Discards access_scope: access is enforced in the application layer
            - Operations must be SYNC callables

        For PostgreSQL:
            - Acquires connection from pool
            - Begins transaction via asyncpg transaction context manager
            - Binds access_scope through the session profile before the body runs
            - Commits on context exit, rollbacks on exception
            - Operations must be ASYNC callables

        Raises:
            RuntimeError: If backend is shut down or circuit breaker is open

        Example (Embedding-First Pattern):
            # Generate embedding OUTSIDE transaction (may be slow/fail)
            embedding = await embedding_provider.embed_query(text)

            # All DB operations in single atomic transaction
            async with backend.begin_transaction(access_scope=scope) as txn:
                context_id = await _store_context(txn, text, metadata)
                await _store_tags(txn, context_id, tags)
                await _store_embedding(txn, context_id, embedding)
                # COMMIT happens here only if ALL succeed

        Example (Rollback Scenario):
            async with backend.begin_transaction(access_scope=scope) as txn:
                context_id = await _store_context(txn, ...)
                await _store_tags(txn, context_id, tags)
                # If this raises, context + tags are rolled back
                await _store_embedding(txn, context_id, embedding)
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """
        Get backend health metrics and statistics.

        Returns a dictionary containing backend-specific health metrics,
        connection pool statistics, and performance indicators.

        The mapping is published to MCP clients verbatim as ``connection_metrics``
        (app/tools/discovery.py), and its declared shape is
        ``ConnectionMetricsDict`` in app/discovery_types.py, which FastMCP turns into the
        advertised ``get_statistics`` outputSchema. A key an implementation emits
        but that TypedDict does not declare is DROPPED for every client consuming
        the typed output, so a new backend must either restrict itself to the
        keys below or extend ``ConnectionMetricsDict`` in the same change.

        Returns:
            Dictionary with metrics. The cross-backend contract guarantees exactly
            two keys, so a client can identify the backend and read one pool bound
            without branching:
                - backend_type: str - Backend identifier ('sqlite', 'postgresql')
                - pool_size: int - A pool bound whose meaning is backend-specific
                  (PostgreSQL: the pool's CURRENT size, present only once the pool
                  exists; SQLite: the reader-pool BOUND, since writes are
                  serialized onto one writer connection)

        Keys the shipped implementations add on top of that contract, all declared
        in ``ConnectionMetricsDict``:

        SQLite (SQLiteBackend.get_metrics, all always present):
            total_connections, active_connections, failed_connections,
            total_queries, failed_queries, write_queue_size, circuit_state,
            consecutive_failures, last_error, last_error_time

        PostgreSQL (PostgreSQLBackend.get_metrics):
            always: total_queries, failed_queries, last_error, last_error_time,
                circuit_state, consecutive_failures
            only while a pool exists: pool_size, pool_idle, pool_min_size,
                pool_max_size
            only after detection ran: pgpool_detected, pgpool_version,
                session_mode_pooler_detected

        Example:
            metrics = backend.get_metrics()
            print(f"Backend: {metrics['backend_type']}")
            print(f"Circuit state: {metrics.get('circuit_state', 'unknown')}")
        """
        ...

    @property
    def backend_type(self) -> str:
        """
        Get the backend type identifier.

        Returns:
            Backend type string: 'sqlite' or 'postgresql'

        This property is used for:
        - Logging and monitoring
        - Conditional SQL generation in repositories
        - Backend-specific optimizations
        - Metrics tagging

        Example:
            if backend.backend_type == 'sqlite':
                # Use SQLite-specific query
                query = 'SELECT json_extract(metadata, "$.status") FROM ...'
            elif backend.backend_type == 'postgresql':
                # Use PostgreSQL-specific query
                query = "SELECT metadata->>'status' FROM ..."
        """
        ...

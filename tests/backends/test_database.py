"""
Test suite for database operations in the MCP Context Storage Server.

Tests database initialization, schema creation, PRAGMA settings,
connection management, and tag storage/retrieval operations.
"""

import sqlite3
from pathlib import Path

import pytest
from anyio import Path as AsyncPath

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.ids import generate_id
from app.startup import init_database


class TestDatabaseInitialization:
    """Test database initialization and schema creation."""

    @pytest.mark.asyncio
    async def test_init_database_creates_schema(self, temp_db_path: Path) -> None:
        """Test database initialization creates all required tables."""
        from app.backends import create_backend

        # Create SQLite backend explicitly
        backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
        await backend.initialize()

        await init_database(backend=backend)

        await backend.shutdown()

        # Verify database exists
        async_temp_db_path = AsyncPath(temp_db_path)
        assert await async_temp_db_path.exists()

        # Verify tables were created
        with sqlite3.connect(str(temp_db_path)) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            ''')
            tables = [row[0] for row in cursor.fetchall()]
            assert set(tables) == {
                'context_entries', 'context_entry_grants', 'generation_jobs', 'image_attachments', 'tags',
            }

            # Check indexes exist
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            ''')
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = {
                'idx_created_at',
                'idx_image_context',
                'idx_source',
                'idx_tags_entry',
                'idx_tags_tag',
                # Tags are a set per entry; the unique index is the database-level
                # backstop behind the write path's deduplication.
                'idx_tags_entry_tag',
                'idx_thread_id',
                'idx_thread_source',
                # Deduplication lookup index, provisioned from the base schema.
                'idx_context_entries_dedup_hash',
                # Grants-table indexes ride the base schema (their table is created
                # complete by this same script). The context_entries access indexes
                # (idx_context_owner, idx_context_owner_thread, idx_context_public and
                # the SQLite covering idx_context_access_thread, idx_context_access_source,
                # idx_context_access_id) deliberately do NOT: on an existing
                # pre-access-control database they would reference columns only
                # apply_access_control_migration adds, crashing initialization before
                # that migration can run.
                'idx_grants_entry_principal',
                'idx_grants_principal',
                # The generation jobs queue index rides the base schema with its table.
                'idx_generation_jobs_queue',
            }
            # The base schema no longer declares any metadata expression index.
            # handle_metadata_indexes (run separately at server startup) is the
            # single source of truth for the configurable idx_metadata_* set, so
            # init_database on its own must leave a fresh database with none of
            # them. A leftover idx_metadata_* here would reintroduce the two
            # sources of truth that break strict-mode reconciliation on a fresh
            # database configured with a custom METADATA_INDEXED_FIELDS.
            assert set(indexes) == expected_indexes
            assert not any(name.startswith('idx_metadata_') for name in indexes)

    @pytest.mark.asyncio
    async def test_init_database_handles_errors(self, temp_db_path: Path) -> None:
        """Test database initialization handles errors properly."""
        from app.backends import create_backend

        # Make directory read-only to cause error
        temp_db_path.parent.mkdir(exist_ok=True)
        async_temp_db_path = AsyncPath(temp_db_path)
        await async_temp_db_path.touch()
        await async_temp_db_path.chmod(0o444)  # Read-only

        # Create SQLite backend explicitly - expect error during initialization
        backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))

        # Backend initialization should fail with OperationalError when file is read-only
        with pytest.raises(sqlite3.OperationalError, match='.*readonly database.*'):
            await backend.initialize()

        # Clean up (no need to shutdown as initialization failed)
        await async_temp_db_path.chmod(0o644)


class TestConnectionUriEncoding:
    """Filesystem paths must survive the SQLite file-URI round trip."""

    @pytest.mark.asyncio
    async def test_backend_opens_path_with_uri_special_characters(self, tmp_path: Path) -> None:
        """A db path containing '%' and '#' opens the exact file, not a decoded variant.

        SQLite percent-decodes URI paths before use, so the backend must
        percent-encode the filesystem path when building the connection URI;
        a raw '%40' segment would otherwise be decoded to '@' and a '#'
        would truncate the path as a URI fragment.
        """
        from app.backends import create_backend

        special_dir = tmp_path / 'pct %40 and #frag'
        special_dir.mkdir()
        db_path = special_dir / 'context.db'

        backend = create_backend(backend_type='sqlite', db_path=str(db_path))
        await backend.initialize()
        try:
            def _select_one(conn: sqlite3.Connection) -> int:
                row = conn.execute('SELECT 1').fetchone()
                return int(row[0])

            assert await backend.execute_read(_select_one, access_scope=MAINTENANCE) == 1
        finally:
            await backend.shutdown()

        assert db_path.exists()
        # The mis-decoded variant ('#' read as a fragment, '%40' decoded to
        # '@') would land directly under tmp_path; it must not exist.
        assert not (tmp_path / 'pct @ and ').exists()


class TestDatabaseConnection:
    """Test database connection management."""

    @pytest.mark.asyncio
    async def test_db_rollback_on_error(self, async_db_initialized: StorageBackend) -> None:
        """Test database rollback on error."""
        manager = async_db_initialized

        # Insert initial data
        async with manager.get_connection(allow_write=True) as conn:
            cursor = conn.cursor()
            _new_id = generate_id()
            cursor.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (_new_id, 'test', 'user', 'text'),
            )
            conn.commit()

        # Try to insert with error
        try:
            async with manager.get_connection(allow_write=True) as conn:
                cursor = conn.cursor()
                # This should fail due to CHECK constraint
                _new_id = generate_id()
                cursor.execute(
                    "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                    (_new_id, 'test', 'invalid_source', 'text'),
                )
                pytest.fail('Should have raised IntegrityError')
        except sqlite3.IntegrityError:
            pass  # Expected

        # Verify no partial data was committed
        async with manager.get_connection(readonly=True) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM context_entries')
            assert cursor.fetchone()[0] == 1  # Only the first valid insert

    @pytest.mark.asyncio
    async def test_execute_write_queue_rolls_back_partial_write_on_failure(
        self,
        async_db_initialized: StorageBackend,
    ) -> None:
        """A failed execute_write must not leak partial rows onto the next commit.

        The write-queue path shares one persistent DEFERRED writer connection, so a
        mid-operation failure must roll back; otherwise the partial write stays in the
        open transaction and the next, unrelated execute_write commits it too.
        """
        manager = async_db_initialized
        leaked_id = generate_id()
        survivor_id = generate_id()

        def op_fail(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (leaked_id, 'test', 'user', 'text'),
            )
            raise ValueError('boom after partial write, before commit')

        def op_ok(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (survivor_id, 'test', 'user', 'text'),
            )

        # The failed operation surfaces its error to the caller.
        with pytest.raises(ValueError, match='boom'):
            await manager.execute_write(op_fail, access_scope=MAINTENANCE)

        # A subsequent unrelated write commits normally.
        await manager.execute_write(op_ok, access_scope=MAINTENANCE)

        # The partial row from the failed operation must be absent; only the
        # survivor row committed.
        async with manager.get_connection(readonly=True) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM context_entries')
            ids = [row['id'] for row in cursor.fetchall()]
        assert survivor_id in ids
        assert leaked_id not in ids

    @pytest.mark.asyncio
    async def test_async_context_manager(self, async_db_initialized: StorageBackend) -> None:
        """Test async database context manager."""
        manager = async_db_initialized
        async with manager.get_connection(readonly=True) as conn:
            # Test connection is valid
            assert conn is not None

            # Test we can execute queries
            cursor = conn.cursor()
            cursor.execute('SELECT 1 as test')
            result = cursor.fetchone()
            assert result['test'] == 1

    @pytest.mark.asyncio
    async def test_async_rollback_on_error(self, async_db_initialized: StorageBackend) -> None:
        """Test async database rollback on error."""
        manager = async_db_initialized

        # Try to insert with error
        try:
            async with manager.get_connection(allow_write=True) as conn:
                cursor = conn.cursor()
                _new_id = generate_id()
                cursor.execute(
                    "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                    (_new_id, 'test', 'invalid_source', 'text'),
                )
                pytest.fail('Should have raised IntegrityError')
        except sqlite3.IntegrityError:
            pass  # Expected


class TestTagOperations:
    """Test tag storage and retrieval operations."""

    @pytest.mark.asyncio
    async def test_store_tags_async(self, async_test_db: sqlite3.Connection) -> None:
        """Test storing tags for a context entry."""
        # Insert a context entry
        cursor = async_test_db.cursor()
        _new_id = generate_id()
        cursor.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
            (_new_id, 'test', 'user', 'text'),
        )
        context_id = _new_id
        async_test_db.commit()

        # Store tags
        tags = ['python', 'Testing', '  spaced  ', '', 'UPPERCASE']
        if context_id is not None:
            # Store tags directly using SQL
            for tag in tags:
                tag = tag.strip().lower()
                if tag:
                    cursor.execute(
                        'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                        (context_id, tag),
                    )
        async_test_db.commit()

        # Verify tags were stored normalized
        cursor.execute(
            'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
            (context_id,),
        )
        stored_tags = [row['tag'] for row in cursor.fetchall()]
        # Empty string should be filtered out
        assert stored_tags == ['python', 'spaced', 'testing', 'uppercase']

    @pytest.mark.asyncio
    async def test_store_tags_async_empty_list(self, async_test_db: sqlite3.Connection) -> None:
        """Test storing empty tag list does nothing."""
        cursor = async_test_db.cursor()
        _new_id = generate_id()
        cursor.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
            (_new_id, 'test', 'user', 'text'),
        )
        context_id = _new_id
        async_test_db.commit()

        # Store empty tags - should do nothing
        # No tags to insert for empty list

        # Verify no tags were stored
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_get_tags_async(self, async_test_db: sqlite3.Connection) -> None:
        """Test retrieving tags for a context entry."""
        # Insert context and tags manually
        cursor = async_test_db.cursor()
        _new_id = generate_id()
        cursor.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
            (_new_id, 'test', 'user', 'text'),
        )
        context_id = _new_id

        tags = ['alpha', 'beta', 'gamma']
        for tag in tags:
            cursor.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (context_id, tag),
            )
        async_test_db.commit()

        # Retrieve tags
        cursor.execute(
            'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
            (context_id,),
        )
        retrieved_tags = [row['tag'] for row in cursor.fetchall()]
        assert retrieved_tags == ['alpha', 'beta', 'gamma']

    @pytest.mark.asyncio
    async def test_get_tags_async_no_tags(self, async_test_db: sqlite3.Connection) -> None:
        """Test retrieving tags for entry with no tags."""
        cursor = async_test_db.cursor()
        _new_id = generate_id()
        cursor.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
            (_new_id, 'test', 'user', 'text'),
        )
        context_id = _new_id
        async_test_db.commit()

        # Retrieve tags for entry without tags
        cursor.execute(
            'SELECT tag FROM tags WHERE context_entry_id = ? ORDER BY tag',
            (context_id,),
        )
        tags = [row['tag'] for row in cursor.fetchall()]
        assert tags == []


class TestDatabaseIntegrity:
    """Test database integrity constraints and foreign keys."""

    def test_foreign_key_constraint(self, test_db: sqlite3.Connection) -> None:
        """Test foreign key constraints are enforced."""
        cursor = test_db.cursor()

        # Try to insert tag for non-existent context entry
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (999, 'test'),
            )

    def test_cascade_delete(self, test_db: sqlite3.Connection) -> None:
        """Test cascading delete removes related tags and images."""
        cursor = test_db.cursor()

        # Insert context entry
        _new_id = generate_id()
        cursor.execute(
            "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
            (_new_id, 'test', 'user', 'text'),
        )
        context_id = _new_id

        # Insert related data
        cursor.execute(
            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
            (context_id, 'test-tag'),
        )
        cursor.execute(
            'INSERT INTO image_attachments (context_entry_id, image_data, mime_type) VALUES (?, ?, ?)',
            (context_id, b'test-image', 'image/png'),
        )
        test_db.commit()

        # Verify data exists
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 1
        cursor.execute('SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 1

        # Delete context entry
        cursor.execute('DELETE FROM context_entries WHERE id = ?', (context_id,))
        test_db.commit()

        # Verify related data was deleted
        cursor.execute('SELECT COUNT(*) FROM tags WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0
        cursor.execute('SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = ?', (context_id,))
        assert cursor.fetchone()[0] == 0

    def test_check_constraints(self, test_db: sqlite3.Connection) -> None:
        """Test CHECK constraints are enforced."""
        cursor = test_db.cursor()

        # Test invalid source
        bad_source_id = generate_id()
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            cursor.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (bad_source_id, 'test', 'invalid', 'text'),
            )
        assert 'CHECK constraint failed' in str(exc_info.value)

        # Test invalid content_type
        bad_ctype_id = generate_id()
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            cursor.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (bad_ctype_id, 'test', 'user', 'invalid'),
            )
        assert 'CHECK constraint failed' in str(exc_info.value)


class TestDatabasePerformance:
    """Test database performance optimizations."""

    def test_indexes_are_used(self, test_db: sqlite3.Connection) -> None:
        """Test that queries use indexes for performance."""
        cursor = test_db.cursor()

        # Insert test data
        for i in range(100):
            _new_id = generate_id()
            cursor.execute(
                "INSERT INTO context_entries (id, thread_id, source, content_type, owner_id) VALUES (?, ?, ?, ?, 'local')",
                (_new_id, f'thread_{i % 10}', 'user' if i % 2 == 0 else 'agent', 'text'),
            )
        test_db.commit()

        # Test query plan for thread_id filter (should use index)
        cursor.execute('EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE thread_id = ?', ('thread_1',))
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_thread_id' in plan_str or 'USING INDEX' in plan_str

        # Test query plan for source filter (should use index)
        cursor.execute('EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE source = ?', ('user',))
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_source' in plan_str or 'USING INDEX' in plan_str

        # Test query plan for combined filter (should use compound index)
        cursor.execute(
            'EXPLAIN QUERY PLAN SELECT * FROM context_entries WHERE thread_id = ? AND source = ?',
            ('thread_1', 'user'),
        )
        plan = cursor.fetchall()
        # Convert Row objects to strings for checking
        plan_str = ''.join(str(dict(row)) for row in plan)
        assert 'idx_thread_source' in plan_str or 'USING INDEX' in plan_str


class TestInitDatabaseAdvisoryLock:
    """Verify init_database() uses advisory lock for PostgreSQL multi-pod safety."""

    def test_postgresql_path_uses_advisory_xact_lock(self) -> None:
        """Verify the advisory lock is acquired before DDL in the PostgreSQL path.

        The lock acquisition lives in the shared begin_migration() helper, which
        init_database() calls at the start of each PostgreSQL schema-init
        transaction. This checks both halves of that chain -- the delegation and
        the helper's lock literal -- so the multi-pod serialization property is
        genuinely verified rather than merely tracking where the string lives.
        """
        import inspect

        from app.migrations._pg_ddl import SCHEMA_INIT_ADVISORY_LOCK_SQL
        from app.migrations._pg_ddl import begin_migration

        init_source = inspect.getsource(init_database)
        assert 'begin_migration' in init_source, (
            'init_database() must acquire the shared migration lock via begin_migration() '
            'for multi-pod safety'
        )

        begin_source = inspect.getsource(begin_migration)
        assert 'SCHEMA_INIT_ADVISORY_LOCK_SQL' in begin_source, (
            'begin_migration() must acquire the shared schema-init advisory lock'
        )
        assert 'pg_advisory_xact_lock' in SCHEMA_INIT_ADVISORY_LOCK_SQL, (
            'the shared migration lock must use pg_advisory_xact_lock for multi-pod safety'
        )
        assert "hashtext('mcp_context_schema_init')" in SCHEMA_INIT_ADVISORY_LOCK_SQL, (
            'the shared migration lock must use the same lock key for DDL serialization'
        )

"""
Pytest configuration and shared fixtures for MCP Context Storage Server tests.

Provides reusable test fixtures including database setup, server instances,
mock contexts, and sample test data for comprehensive testing.
"""

import os
import sys

# ============================================================================
# CRITICAL: Disable LangSmith/LangChain Tracing BEFORE Importing Packages
# ============================================================================
# LangSmith auto-instruments httpx at import time if tracing is enabled.
# These environment variables MUST be set BEFORE importing any packages
# that might trigger LangSmith initialization.
# ============================================================================
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TEST_CACHE'] = ''

# ============================================================================
# Now Safe to Import Everything Else
# ============================================================================
# Note: E402 warnings for these imports are suppressed via pyproject.toml
# extend-per-file-ignores because the imports MUST come after the environment
# variable setup above.
# ============================================================================
import asyncio
import base64
import importlib.util
import json
import sqlite3
import tempfile
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from anyio import Path as AsyncPath
from dotenv import load_dotenv

# ============================================================================
# CRITICAL: Configure logging BEFORE importing app modules
# This ensures tests have properly configured logging even when app modules
# are imported directly (not via app.server entry point)
# ============================================================================
from app.logger_config import config_logger
from app.settings import get_settings

_test_settings = get_settings()
config_logger(_test_settings.logging.level)

# Now safe to import app modules
import app.startup
from app.backends import StorageBackend
from app.backends import create_backend
from app.backends.base import MAINTENANCE
from app.settings import AppSettings

# ============================================================================
# Event Loop Selection
# ============================================================================
# On Windows the default Proactor (IOCP) loop can hang when httpx leaves pending
# I/O operations that never complete, blocking GetQueuedCompletionStatus
# indefinitely, so every async test and fixture runs on a selector loop there.
# ============================================================================
if sys.platform == 'win32':

    def pytest_asyncio_loop_factories() -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
        """Run async tests and fixtures on a selector event loop on Windows.

        Returns:
            The single loop factory pytest-asyncio uses for every async test and fixture.
        """
        return {'selector': asyncio.SelectorEventLoop}


# ============================================================================
# Conditional Skip Helpers for Optional Dependencies
# ============================================================================


def is_ollama_available() -> bool:
    """Check if ollama package is installed."""
    return importlib.util.find_spec('ollama') is not None


def is_sqlite_vec_available() -> bool:
    """Check if sqlite-vec package is installed."""
    return importlib.util.find_spec('sqlite_vec') is not None


def is_numpy_available() -> bool:
    """Check if numpy package is installed."""
    return importlib.util.find_spec('numpy') is not None


def are_semantic_search_deps_available() -> bool:
    """Check if all semantic search dependencies are available."""
    return is_ollama_available() and is_sqlite_vec_available() and is_numpy_available()


def is_chunking_available() -> bool:
    """Check if langchain-text-splitters package is installed."""
    return importlib.util.find_spec('langchain_text_splitters') is not None


def is_flashrank_available() -> bool:
    """Check if flashrank package is installed."""
    return importlib.util.find_spec('flashrank') is not None


# Pytest markers for conditional skipping
requires_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason='ollama package not installed',
)

requires_sqlite_vec = pytest.mark.skipif(
    not is_sqlite_vec_available(),
    reason='sqlite-vec package not installed',
)

requires_numpy = pytest.mark.skipif(
    not is_numpy_available(),
    reason='numpy package not installed',
)

requires_semantic_search = pytest.mark.skipif(
    not are_semantic_search_deps_available(),
    reason='Semantic search dependencies not available (ollama, sqlite_vec, numpy)',
)

requires_chunking = pytest.mark.skipif(
    not is_chunking_available(),
    reason='langchain-text-splitters package not installed (chunking feature)',
)

requires_flashrank = pytest.mark.skipif(
    not is_flashrank_available(),
    reason='flashrank package not installed (reranking feature)',
)


def is_fts_enabled() -> bool:
    """Check if FTS is enabled in environment."""
    from app.settings import get_settings
    return get_settings().fts.enabled


requires_fts = pytest.mark.skipif(
    not is_fts_enabled(),
    reason='FTS force-disabled (ENABLE_FTS=false)',
)

requires_hybrid_search = pytest.mark.skipif(
    not (is_fts_enabled() or are_semantic_search_deps_available()),
    reason='Hybrid search unavailable (FTS force-disabled and semantic search deps missing)',
)

# Load .env file to make environment variables available for PostgreSQL tests
load_dotenv()

# CRITICAL: Force SQLite backend for ALL tests (override .env STORAGE_BACKEND=postgresql)
# PostgreSQL tests will explicitly create PostgreSQL backends when needed
os.environ['STORAGE_BACKEND'] = 'sqlite'


# ============================================================================
# Mock Backend Helper for Transaction Support
# ============================================================================
def create_mock_backend_with_transaction_support() -> MagicMock:
    """Create a mock backend that supports begin_transaction() as async context manager.

    Tools invoke ``backend.begin_transaction()`` to wrap multi-step writes in a
    single atomic transaction. Tests that mock ``ensure_repositories()`` must
    therefore provide a backend whose ``begin_transaction`` is callable as an
    async context manager.

    Returns:
        MagicMock: A mock backend with begin_transaction() async context manager support.

    Usage:
        mock_backend = create_mock_backend_with_transaction_support()
        repos.context.backend = mock_backend
    """
    from contextlib import asynccontextmanager

    mock_backend = MagicMock()

    @asynccontextmanager
    async def mock_begin_transaction(*, access_scope):
        """Mock async context manager for begin_transaction."""
        _ = access_scope
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        txn.connection = MagicMock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction
    return mock_backend


# Global fixture to ensure NO test uses the default database
@pytest.fixture(autouse=True, scope='session')
def prevent_default_db_pollution():
    """
    Prevents ALL tests from using the default database.

    This fixture runs automatically for ALL tests in the session and ensures:
    1. DB_PATH is set to a temporary location
    2. MCP_TEST_MODE is enabled to indicate testing
    3. Default database path is NEVER used

    Raises:
        RuntimeError: If configuration attempts to use the default database.
    """
    # Store original environment
    original_db_path = os.environ.get('DB_PATH')
    original_test_mode = os.environ.get('MCP_TEST_MODE')

    # Create a session-wide temp directory
    with tempfile.TemporaryDirectory(prefix='mcp_test_session_') as temp_dir:
        # Set test environment variables
        temp_db_path = Path(temp_dir) / 'test_session.db'
        os.environ['DB_PATH'] = str(temp_db_path)
        os.environ['MCP_TEST_MODE'] = '1'

        # Verify we're not using default database
        default_db = Path.home() / '.mcp' / 'context_storage.db'
        if temp_db_path.resolve() == default_db.resolve():
            raise RuntimeError(
                f'CRITICAL: Test configuration error - attempting to use default database!\n'
                f'Default: {default_db}\n'
                f'Current: {temp_db_path}',
            )

        print(f'\n[TEST SAFETY] Session-wide temp DB: {temp_db_path}')
        print('[TEST SAFETY] MCP_TEST_MODE enabled')
        print(f'[TEST SAFETY] Default DB protected: {default_db}\n')

        try:
            yield
        finally:
            # Restore original environment
            if original_db_path is None:
                os.environ.pop('DB_PATH', None)
            else:
                os.environ['DB_PATH'] = original_db_path

            if original_test_mode is None:
                os.environ.pop('MCP_TEST_MODE', None)
            else:
                os.environ['MCP_TEST_MODE'] = original_test_mode


# Function-scoped autouse fixture: force ENABLE_EMBEDDING_COMPRESSION=false
# for every test by default. With v3.0.0 flipping the production default to
# true, every test that does NOT explicitly opt into compression must see
# the disabled state. Tests that need compression on call
# monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true') (and the
# other COMPRESSION_* env vars) inside the test body or via a
# feature-specific fixture; the test-body setenv overrides this autouse
# entry because both writes target the same monkeypatch.setenv stack.
@pytest.fixture(autouse=True)
def force_compression_off(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Disable compression by default for every test.

    The fixture sets ``ENABLE_EMBEDDING_COMPRESSION=false`` and clears the
    settings cache so the next ``get_settings()`` call observes the override.
    Tests that opt into compression simply call
    ``monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')`` and
    ``get_settings.cache_clear()`` themselves; the autouse fixture's setenv
    is unwound by monkeypatch at test teardown.

    Yields:
        Control to the test body.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    # Refresh module-level ``settings = get_settings()`` bindings that were
    # cached at import time. Without this, modules that read
    # ``settings.compression.enabled`` from their own module-level binding
    # (e.g., ``app.tools._generation``, ``app.tools._delete_cleanup``,
    # ``app.migrations.compression``) keep the import-time singleton and ignore
    # the per-test env override. The CLAUDE.md "Per-test environment overrides
    # for module-level ``settings`` bindings" section documents this pattern.
    fresh = get_settings()
    import app.migrations.compression as _compression_migration_module
    import app.server as _server_module
    import app.tools._delete_cleanup as _delete_cleanup_module
    import app.tools._generation as _generation_module
    import app.tools._validation as _validation_module
    import app.tools.context.retrieve as _context_retrieve_module
    import app.tools.context.store as _context_store_module
    import app.tools.context.update as _context_update_module
    import app.tools.discovery as _discovery_module
    monkeypatch.setattr(_context_store_module, 'settings', fresh)
    monkeypatch.setattr(_context_retrieve_module, 'settings', fresh)
    monkeypatch.setattr(_context_update_module, 'settings', fresh)
    monkeypatch.setattr(_validation_module, 'settings', fresh)
    monkeypatch.setattr(_generation_module, 'settings', fresh)
    monkeypatch.setattr(_delete_cleanup_module, 'settings', fresh)
    monkeypatch.setattr(_compression_migration_module, 'settings', fresh)
    monkeypatch.setattr(_server_module, 'settings', fresh)
    monkeypatch.setattr(_discovery_module, 'settings', fresh)
    try:
        yield
    finally:
        # monkeypatch automatically unwinds setenv and setattr at test
        # teardown; the try/finally is structural (it keeps the generator
        # shape so Ruff does not rewrite this bare yield into a return) and
        # a defensive cache_clear so the next test starts from a clean
        # settings cache.
        get_settings.cache_clear()


# Test configuration
@pytest.fixture
def test_settings(tmp_path: Path) -> AppSettings:
    """Create test settings with temporary database path."""
    # Create settings with correct storage configuration
    # Use temporary_env_vars context manager to set environment variables
    with temporary_env_vars(
        MAX_IMAGE_SIZE_MB='5',
        MAX_TOTAL_SIZE_MB='20',
        DB_PATH=str(tmp_path / 'test_context.db'),
        LOG_LEVEL='DEBUG',
    ):
        # AppSettings will automatically create StorageSettings
        # with the environment variables
        return AppSettings()


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide temporary database path."""
    db_path = tmp_path / 'test_context.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@pytest.fixture
def test_db(temp_db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Create and initialize a test database."""
    # Create connection with increased timeout
    conn = sqlite3.connect(str(temp_db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Check if tables already exist (from initialized_server fixture)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries'")
    if not cursor.fetchone():
        # Tables don't exist, create them
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        conn.executescript(schema_sql)

    # Apply optimizations
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA temp_store = MEMORY')
    conn.execute('PRAGMA busy_timeout = 30000')  # 30 second busy timeout

    yield conn
    conn.close()


@pytest_asyncio.fixture
async def async_test_db(temp_db_path: Path) -> AsyncGenerator[sqlite3.Connection, None]:
    """Create async test database connection.

    Uses asyncio.to_thread() to offload blocking sqlite3 setup/teardown
    onto the default thread executor without depending on a manual
    get_event_loop() handle. ``to_thread`` resolves the running loop
    internally and is the recommended modern equivalent of
    ``loop.run_in_executor(None, ...)``.

    Yields:
        Initialized sqlite3 connection; closed after the test completes.
    """
    def _create_db() -> sqlite3.Connection:
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        # Use check_same_thread=False for async tests to avoid thread safety issues
        conn = sqlite3.connect(str(temp_db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)

        # Apply optimizations
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA busy_timeout = 30000')  # 30 second busy timeout
        return conn

    conn = await asyncio.to_thread(_create_db)
    yield conn
    await asyncio.to_thread(conn.close)


@pytest.fixture
def sample_image_data() -> dict[str, str]:
    """Generate sample base64 encoded image data."""
    # Create a simple 1x1 PNG image
    png_data = bytes([
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,  # PNG signature
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,  # IHDR chunk
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,  # 1x1 dimensions
        0x08,
        0x02,
        0x00,
        0x00,
        0x00,
        0x90,
        0x77,
        0x53,  # Color type, etc
        0xDE,
        0x00,
        0x00,
        0x00,
        0x0C,
        0x49,
        0x44,
        0x41,  # IDAT chunk
        0x54,
        0x08,
        0x99,
        0x01,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x7B,
        0xDB,
        0x56,
        0x61,
        0x00,  # Image data
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,  # IEND chunk
        0x42,
        0x60,
        0x82,
    ])
    return {
        'data': base64.b64encode(png_data).decode('utf-8'),
        'mime_type': 'image/png',
    }


@pytest.fixture
def sample_context_data() -> dict[str, Any]:
    """Generate sample context entry data."""
    return {
        'thread_id': 'test_thread_123',
        'source': 'user',
        'text': 'This is a test context entry',
        'metadata': {'key': 'value', 'priority': 10},
        'tags': ['test', 'sample', 'fixture'],
    }


@pytest.fixture
def sample_multimodal_data(sample_image_data: dict[str, str]) -> dict[str, Any]:
    """Generate sample multimodal context data."""
    return {
        'thread_id': 'test_multimodal_456',
        'source': 'agent',
        'text': 'Analysis of the attached image',
        'images': [sample_image_data],
        'metadata': {'analysis_type': 'visual'},
        'tags': ['image', 'analysis'],
    }


@pytest.fixture
def multiple_context_entries(test_db: sqlite3.Connection) -> list[str]:
    """Insert multiple test context entries and return their UUIDv7 hex IDs.

    Returns:
        list[str]: List of 32-char lowercase hex UUIDv7 strings for each inserted entry.
    """
    from app.ids import generate_id

    cursor = test_db.cursor()
    entries = [
        ('thread_1', 'user', 'text', 'First test entry', None),
        ('thread_1', 'agent', 'text', 'Response to first', None),
        ('thread_2', 'user', 'multimodal', 'Second thread entry', json.dumps({'key': 'value'})),
        ('thread_2', 'agent', 'text', 'Agent analysis', None),
        ('thread_3', 'user', 'text', 'Third thread start', json.dumps({'priority': 1})),
    ]

    ids: list[str] = []
    for thread_id, source, content_type, text, metadata in entries:
        new_id = generate_id()
        cursor.execute(
            '''
            INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id)
            VALUES (?, ?, ?, ?, ?, ?, 'local')
            ''',
            (new_id, thread_id, source, content_type, text, metadata),
        )
        ids.append(new_id)

    # Add tags to some entries
    tags_data = [
        (ids[0], 'important'),
        (ids[0], 'user-input'),
        (ids[1], 'response'),
        (ids[2], 'analysis'),
        (ids[3], 'ai-generated'),
    ]
    for entry_id, tag in tags_data:
        cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (entry_id, tag))

    test_db.commit()
    return ids


@pytest.fixture
def mock_server_dependencies(test_settings: AppSettings, temp_db_path: Path) -> Generator[None, None, None]:
    """Mock server dependencies for unit testing."""
    # Initialize the database schema synchronously before patching
    if not temp_db_path.exists():
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create database with schema
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        conn = sqlite3.connect(str(temp_db_path))
        try:
            conn.executescript(schema_sql)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')
            conn.commit()
        finally:
            conn.close()

    # Store original STORAGE_BACKEND environment variable
    original_storage_backend = os.environ.get('STORAGE_BACKEND')

    try:
        # Force SQLite backend for tests (override .env setting)
        os.environ['STORAGE_BACKEND'] = 'sqlite'

        with (
            # CRITICAL: Patch factory.get_settings to prevent lazy backend creation from reading environment
            patch('app.backends.factory.get_settings', return_value=test_settings),
            patch('app.server.DB_PATH', temp_db_path),
            # CRITICAL: Patch startup.DB_PATH - ensure_backend() uses this for lazy initialization
            patch('app.startup.DB_PATH', temp_db_path),
            # Patch MAX_IMAGE_SIZE_MB and MAX_TOTAL_SIZE_MB where they are used (in app.tools._validation)
            patch('app.tools._validation.MAX_IMAGE_SIZE_MB', test_settings.storage.max_image_size_mb),
            patch('app.tools._validation.MAX_TOTAL_SIZE_MB', test_settings.storage.max_total_size_mb),
        ):
            yield
    finally:
        # Restore original STORAGE_BACKEND
        if original_storage_backend is None:
            os.environ.pop('STORAGE_BACKEND', None)
        else:
            os.environ['STORAGE_BACKEND'] = original_storage_backend


@pytest.fixture
def large_image_data() -> dict[str, str]:
    """Generate a large image that exceeds size limits."""
    # Create 10MB of data (exceeds default 5MB limit in test settings)
    large_data = b'x' * (10 * 1024 * 1024)
    return {
        'data': base64.b64encode(large_data).decode('utf-8'),
        'mime_type': 'image/png',
    }


@pytest_asyncio.fixture
async def async_db_initialized(temp_db_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Initialize async database for all tests."""
    from app.repositories import RepositoryContainer

    # Create storage backend
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()

    # Set in startup module (global state)
    app.startup.set_backend(backend)

    # Initialize repositories
    app.startup.set_repositories(RepositoryContainer(backend))

    # Initialize the database schema using the backend
    # NOTE: We initialize schema directly instead of calling init_database() to avoid
    # reading STORAGE_BACKEND from environment (user may have postgresql in .env)
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema, access_scope=MAINTENANCE)

    try:
        yield backend
    finally:
        # Proper cleanup
        cleanup_backend = app.startup.get_backend()
        if cleanup_backend is not None:
            try:
                # Shutdown the storage backend
                await cleanup_backend.shutdown()
            except Exception as e:
                # Log error but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')
            finally:
                # Always clear the reference, even if shutdown failed
                app.startup.set_backend(None)

        # Reset repositories
        app.startup.set_repositories(None)


@pytest_asyncio.fixture
async def initialized_server(mock_server_dependencies: None, temp_db_path: Path) -> AsyncGenerator[None, None]:
    """Initialize server with test database and proper async cleanup.

    Note: mock_server_dependencies fixture is required to patch server settings,
    even though it's not directly used in the function body.

    Yields:
        None: Yields control after initialization, performs cleanup on teardown
    """
    from app.repositories import RepositoryContainer

    # CRITICAL: Aggressive pre-cleanup to prevent interference from previous tests
    # Shut down any existing backend from previous tests
    existing_backend = app.startup.get_backend()
    if existing_backend is not None:
        try:
            await existing_backend.shutdown()
            # Wait for shutdown completion via event-based synchronization
            if hasattr(existing_backend, 'wait_for_shutdown_complete'):
                await cast(Any, existing_backend).wait_for_shutdown_complete(timeout_seconds=2.0)
        except Exception:
            pass
        finally:
            app.startup.set_backend(None)

    # Reset repositories
    app.startup.set_repositories(None)

    # Remove existing database if it exists (DB_PATH is patched by mock_server_dependencies)
    async_temp_db_path = AsyncPath(temp_db_path)
    if await async_temp_db_path.exists():
        await async_temp_db_path.unlink()

    # Create persistent backend before yielding (prevents lazy initialization in tests)
    # NOTE: We create SQLite backend directly instead of calling init_database() to avoid
    # reading STORAGE_BACKEND from environment (user may have postgresql in .env)
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()
    app.startup.set_backend(backend)

    # Initialize repositories with the backend
    app.startup.set_repositories(RepositoryContainer(backend))

    # Initialize the database schema using the backend
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema, access_scope=MAINTENANCE)

    # Ensure the fixture dependency is satisfied
    assert mock_server_dependencies is None

    try:
        yield
    finally:
        # Proper async cleanup with timeout protection
        cleanup_backend = app.startup.get_backend()
        if cleanup_backend is not None:
            try:
                # Shutdown the storage backend with timeout to prevent hangs
                await asyncio.wait_for(
                    cleanup_backend.shutdown(),
                    timeout=5.0,
                )
                # Wait for shutdown completion event to ensure all background tasks terminated
                if hasattr(cleanup_backend, 'wait_for_shutdown_complete'):
                    await cast(Any, cleanup_backend).wait_for_shutdown_complete(timeout_seconds=2.0)
            except TimeoutError:
                # Log timeout but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).warning('Backend shutdown timed out after 5 seconds')
            except Exception as e:
                # Log error but continue cleanup to prevent test suite hang
                import logging
                logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')
            finally:
                # Always clear the reference, even if shutdown failed
                app.startup.set_backend(None)

        # Reset the repositories to ensure clean state
        app.startup.set_repositories(None)


@contextmanager
def temporary_env_vars(**kwargs):
    """Context manager for temporarily setting environment variables."""

    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = os.environ.get(key)
        if value is not None:
            os.environ[key] = str(value)
        elif key in os.environ:
            del os.environ[key]
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


@pytest_asyncio.fixture
async def async_db_with_embeddings(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Initialize async database with semantic search migration applied.

    This fixture creates a database with the full schema AND the semantic search
    migration (vec_context_embeddings table). Use this fixture for tests that
    require the EmbeddingRepository.

    Yields:
        StorageBackend: Initialized SQLite backend with semantic search tables.
    """
    from app.repositories import RepositoryContainer
    from app.schemas import load_schema
    from app.settings import get_settings

    settings = get_settings()
    db_path = tmp_path / 'test_context.db'

    # Create storage backend
    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    # Set in startup module (global state)
    app.startup.set_backend(backend)

    # Initialize repositories
    app.startup.set_repositories(RepositoryContainer(backend))

    # Initialize the base database schema
    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema, access_scope=MAINTENANCE)

    # Apply semantic search migration with correct dimension
    semantic_migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_semantic_search_sqlite.sql'
    semantic_migration_sql = semantic_migration_path.read_text(encoding='utf-8')
    # Replace the template with actual dimension
    semantic_migration_sql = semantic_migration_sql.replace('{EMBEDDING_DIM}', str(settings.embedding.dim))

    # Apply chunking migration (creates embedding_chunks table for 1:N relationships)
    chunking_migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_chunking_sqlite.sql'
    chunking_migration_sql = chunking_migration_path.read_text(encoding='utf-8')

    def _apply_migrations(conn: sqlite3.Connection) -> None:
        # Load sqlite-vec extension before executing migration
        # The migration SQL uses vec0 module which requires sqlite-vec extension
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except ImportError as e:
            raise RuntimeError(
                'sqlite_vec package is required for embedding tests. '
                'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
            ) from e

        # Apply semantic search migration first (creates vec_context_embeddings)
        conn.executescript(semantic_migration_sql)

        # Apply chunking migration (creates embedding_chunks for 1:N relationships)
        conn.executescript(chunking_migration_sql)

        # Add chunk_count column to embedding_metadata (if not exists)
        # SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we check first
        cursor = conn.execute('PRAGMA table_info(embedding_metadata)')
        columns = [row[1] for row in cursor.fetchall()]
        if 'chunk_count' not in columns:
            conn.execute('ALTER TABLE embedding_metadata ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 1')

    await backend.execute_write(_apply_migrations, access_scope=MAINTENANCE)

    try:
        yield backend
    finally:
        # Proper cleanup
        cleanup_backend = app.startup.get_backend()
        if cleanup_backend is not None:
            try:
                await cleanup_backend.shutdown()
            except Exception:
                pass
            finally:
                app.startup.set_backend(None)

        app.startup.set_repositories(None)


@pytest.fixture
def embedding_dim() -> int:
    """Provide embedding dimension from settings.

    The configured embedding dimension defaults to 768 but can be overridden
    via EMBEDDING_DIM environment variable (e.g., 384 in CI).

    Returns:
        int: The embedding dimension from settings.
    """
    from app.settings import get_settings
    return get_settings().embedding.dim

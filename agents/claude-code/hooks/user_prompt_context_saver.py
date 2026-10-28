#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "fastmcp>=4.1.0",
#   "pyyaml",
# ]
# ///
"""
User Prompt Context Saver Hook for Claude Code.

This hook captures user prompts from UserPromptSubmit events and stores them
in the mcp-context-server for enhanced conversation context management.
Pre-built slash commands and prompts matching a configured skip pattern are
not stored; the patterns identify prompts that are not the user's own words,
such as internal hook prompts and the tagged or framed turns Claude Code starts
on the session's behalf.

Note: Currently, the UserPromptSubmit event does not provide images from user
requests, so this hook cannot save image content to the context server. Only
text prompts are captured and stored.

Claude Code kills a hook at the timeout its environment registers and discards
its output, so the hook keeps one shorter time budget of its own,
mcp_client.deadline_seconds: every connection or request timeout is capped by
what remains of the budget, no attempt starts once less than a second remains,
and the run ends with an answer for the model before the budget does. Claude
Code counts its timeout from the moment it starts the hook's command, and uv
prepares the script's environment and starts the interpreter before any hook
code runs, so the budget counts from the launch: the creation of the uv process
that started the hook, recognized by its image name as the hook's parent or,
where the interpreter runs through a venv redirector, as the redirector's
parent. When the hook cannot identify that process (no uv launcher in that
short walk, a process that cannot be read, a parent newer than its child, or a
launch older than the 600 seconds Claude Code gives a command hook by default),
the budget counts from the start of the hook's module code. A startup that has
already used the whole budget answers at once that no store request was sent,
instead of being killed while it stores.

A turn the hook does not store needs only its payload, the configuration, and
the skip patterns, so the hook decides on them first and reads no process
before that decision: the launch is read from the process table, and fastmcp
and asyncio are imported, only for a prompt it stores.

Each run records the stages it reaches in a startup trace, because a run
Claude Code kills leaves no output to say where it stalled. The trace is a file
of its own in the claude_hook_user_prompt_context_saver_traces directory under
the temporary directory, one JSON record per line, each flushed as it is
written; a run that ends normally removes it, so a file that remains belongs
to a run that was killed or crashed, and its last record names the last
stage that run reached. A run keeps at most MAX_KEPT_TRACES traces in the
directory, its own included, and never removes one written within the last
MAX_LAUNCH_AGE_SECONDS, which may belong to a run still in flight.
CLAUDE_HOOK_STARTUP_TRACE set to 0, false, or no turns the trace off. Once the
skip is decided, a run whose trace records reads from the process table when
its interpreter was created and, where a Windows venv redirector started that
interpreter, when the redirector was, and records both before it exits or
stores, so the trace splits the startup into its parts. A creation time does
not change, so reading it late loses nothing, and a run killed before that
read still leaves the stages it reached; a run without a trace never reads it.

A store request that fails or times out after it was sent is never sent again
within the run, because the server may have received it and would store the
message twice; only a failure before the request is sent is retried. When the
message is not confirmed stored, the hook exits 0 and tells the model so
through additionalContext, naming the thread and the stage the run reached:
when no store request was sent, the model stores the prompt itself with the
context-server store tool; when one was sent, the server may still complete
it, so the model searches the thread at the end of the turn and stores a copy
only if none landed, and a chunked store names the chunks the server confirmed.

main() relies on its helpers being correct under the platform contract; only
narrower handlers exist (json.JSONDecodeError for malformed stdin from Claude
Code; an inner except Exception around building the MCP client and its
store_context call to absorb realistic remote-service failures without blocking
the user's workflow, followed by the notice that tells the model the message was
not confirmed stored). There is no outer catch-all except Exception
block: an unexpected exception escapes to Python's default handler, so the hook
exits non-zero and the transcript shows a hook error notice carrying the first
stderr line. The full traceback is recorded only in Claude Code's debug log,
which exists only while its debug logging is on (for example under --debug or
--debug-file); a run with it on shows the underlying code-quality defect so it
can be fixed.

Diagnostic logging is off unless CLAUDE_HOOK_DEBUG_ENABLED is "1", "true", or
"yes". While it is off, the hook writes no log or error file and creates no
directory for one; while it is on, the log location is resolved on the first
write.

Trigger: UserPromptSubmit
"""

import ctypes
import importlib.util
import json
import ntpath
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Protocol
from typing import TextIO
from typing import cast

# The hook's own clock, read on the monotonic and the wall clock as its module code
# starts, after the standard-library imports and before any third-party import. The
# run's budget counts from the earlier launch of the hook whenever
# launcher_clock_start can identify it, and from here otherwise; the startup trace
# counts its elapsed seconds from here.
_HOOK_STARTED_AT = time.monotonic()
_HOOK_STARTED_WALL = time.time()


def _load_config_loader() -> ModuleType:
    """Dynamically load hook_config_loader from the same directory."""
    loader_path = Path(__file__).parent / 'hook_config_loader.py'
    spec = importlib.util.spec_from_file_location('hook_config_loader', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_config_loader from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json_output() -> ModuleType:
    """Dynamically load hook_json_output from the same directory."""
    loader_path = Path(__file__).parent / 'hook_json_output.py'
    spec = importlib.util.spec_from_file_location('hook_json_output', loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load hook_json_output from {loader_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Check if logging is enabled (module-level check, executed once)
_LOGGING_ENABLED = os.environ.get('CLAUDE_HOOK_DEBUG_ENABLED', '').lower() in ('1', 'true', 'yes')

# Message size limits for Windows subprocess pipe buffers
# Windows pipes typically have 64KB buffer, use conservative limits
_MAX_MESSAGE_SIZE = int(os.environ.get('CLAUDE_HOOK_MAX_MESSAGE_SIZE', '32768'))  # 32KB default
_CHUNK_SIZE = int(os.environ.get('CLAUDE_HOOK_CHUNK_SIZE', '30000'))  # 30KB default for chunks
_JSON_OVERHEAD = 500  # Estimated bytes for JSON structure (thread_id, source, etc.)

# Default configuration - used when no config file is provided
DEFAULT_CONFIG: dict[str, Any] = {
    'enabled': True,
    'output_context_id': True,  # Output stored context_id via hookSpecificOutput.additionalContext
    'skip_patterns': [],  # Regex patterns for prompts that are not the user's own words
    'prebuilt_commands': [
        'add-dir',
        'agents',
        'bug',
        'clear',
        'compact',
        'config',
        'cost',
        'doctor',
        'help',
        'init',
        'login',
        'logout',
        'mcp',
        'memory',
        'model',
        'permissions',
        'pr_comments',
        'review',
        'status',
        'terminal-setup',
        'vim',
    ],
    'message_limits': {
        'max_message_size': 32768,
        'chunk_size': 30000,
        'json_overhead': 500,
    },
    'thread_id': {
        'max_retries': 3,
    },
    'mcp_client': {
        'max_retries': 3,
        'max_connection_retries': 3,  # Retry attempts for actual MCP connection
        'connection_timeout': 30.0,  # Timeout per connection attempt (seconds)
        'timeout_first_run': 240.0,
        'timeout_normal': 240.0,
        # The run's whole time budget (seconds), measured from the hook's launch:
        # the creation of the uv process that started it, or the start of the
        # hook's module code when that process cannot be identified. Every timeout
        # above is capped by what remains of it, so the hook answers before the
        # Claude Code hook timeout kills it. A UserPromptSubmit hook gets 30
        # seconds unless its event entry registers a timeout, so this default
        # leaves 10 of them for any shell start before uv, the exit, and a startup
        # the launch read cannot see; a config pairs a longer budget with a
        # longer registered timeout.
        'deadline_seconds': 20.0,
    },
    'mcp_server': {
        # Transport type: 'stdio' (default) or 'http'
        'transport': 'stdio',
        # For stdio transport
        'command': 'uvx',
        'python_version': '3.12',
        'package': 'mcp-context-server[embeddings-ollama]<2.0.0',
        'entry_point': 'mcp-context-server',
        'prewarm_cache': True,  # Pre-warm the uvx cache before connecting over stdio
        # For http transport (used when transport: http)
        # 'url': 'https://mcp-context-server.example.com/mcp',
        # 'headers': {},  # Optional custom headers for authentication
        # 'timeout': 30.0,  # Request timeout in seconds
    },
    'chunking': {
        'fail_mode': 'warn',  # 'silent', 'warn', or 'error'
    },
}


@cache
def _log_file() -> Path:
    """
    Resolve the debug log location on first use, creating its directory.

    Called only while logging is enabled, so a run with logging off creates no
    directory and prints no diagnostic. The result is cached, so each run
    chooses and reports its location once.

    Fallback chain:
    1. CLAUDE_HOOK_DEBUG_FILE environment variable
    2. {CLAUDE_PROJECT_DIR}/.claude/.hook_debug.log
    3. ~/.claude/hook_logs/user_prompt_context_saver.log
    4. {TEMP}/claude_hook_user_prompt_context_saver.log

    Returns:
        Path to the log file (guaranteed to return a valid path)
    """

    def _diagnostic(msg: str) -> None:
        """Report the chosen location, or a fallback that failed, on stderr."""
        with suppress(Exception):
            print(f'[LOG PATH DIAGNOSTIC] {msg}', file=sys.stderr, flush=True)

    # Fallback 1: Explicit debug file location
    debug_file = os.environ.get('CLAUDE_HOOK_DEBUG_FILE')
    if debug_file:
        _diagnostic(f'Using CLAUDE_HOOK_DEBUG_FILE: {debug_file}')
        return Path(debug_file)

    # Fallback 2: Project directory
    project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
    if project_dir:
        claude_dir = Path(project_dir) / '.claude'
        try:
            claude_dir.mkdir(parents=True, exist_ok=True)
            log_path = claude_dir / '.hook_debug.log'
            _diagnostic(f'Using project dir: {log_path}')
            return log_path
        except Exception as e:
            _diagnostic(f'Project dir fallback failed: {e}')

    # Fallback 3: User home directory
    try:
        home = Path.home()
        log_dir = home / '.claude' / 'hook_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'user_prompt_context_saver.log'
        _diagnostic(f'Using home dir: {log_path}')
        return log_path
    except Exception as e:
        _diagnostic(f'Home dir fallback failed: {e}')

    # Fallback 4: System temp directory (always works)
    temp_dir = Path(tempfile.gettempdir())
    log_path = temp_dir / 'claude_hook_user_prompt_context_saver.log'
    _diagnostic(f'Using temp dir: {log_path}')
    return log_path


def log_always(message: str, level: str = 'INFO') -> None:
    """
    Append a timestamped line to the debug log while logging is enabled (never raises).

    Logging is on only when CLAUDE_HOOK_DEBUG_ENABLED is "1", "true", or "yes";
    otherwise this returns at once, before any file or directory is touched.
    While it is on, the first call resolves the log location through the
    fallback chain of _log_file(), and a failed write is swallowed so logging
    never breaks the hook.

    Args:
        message: The message to log
        level: Log level (INFO, ERROR, DEBUG, etc.)
    """
    if not _LOGGING_ENABLED:
        return

    try:
        with _log_file().open('a', encoding='utf-8') as f:
            timestamp = datetime.now(tz=UTC).isoformat()
            f.write(f'{timestamp} [{level}] {message}\n')
    except Exception:
        # Even logging failures are silent - never break the hook
        pass


# Record the run's starting state as the first entries of its log
if _LOGGING_ENABLED:
    log_always('=' * 80)
    log_always('SCRIPT START')
    log_always(f'sys.argv: {sys.argv}')
    log_always(f'cwd: {os.getcwd()}')
    log_always(f'Python version: {sys.version}')
    log_always(f'Python executable: {sys.executable}')
    log_always(f"CLAUDE_PROJECT_DIR: {os.environ.get('CLAUDE_PROJECT_DIR', 'NOT SET')}")
    log_always(f"CLAUDE_HOOK_DEBUG_FILE: {os.environ.get('CLAUDE_HOOK_DEBUG_FILE', 'NOT SET')}")
    log_always(f'Log file location: {_log_file()}')
    log_always(f'stdin isatty: {sys.stdin.isatty()}')


def _warmup_uvx_cache(server_config: dict[str, Any]) -> None:
    """
    Pre-warm uvx cache to avoid cold start delays.

    Runs a lightweight uvx command to ensure the package is cached.
    This helps prevent "Connection closed" errors on first run due to
    uvx downloading packages from PyPI.

    This function:
    - Runs only for the stdio transport, using the loaded configuration,
      because UserPromptSubmit hooks execute under a short Claude Code
      timeout and a warmup subprocess would spend that budget for nothing
      on HTTP deployments
    - Silent failure (never breaks the hook)
    - Uses --help flag for minimal execution time

    Args:
        server_config: The mcp_server section of the loaded configuration
    """
    if not server_config.get('prewarm_cache', True):
        log_always('uvx cache pre-warm disabled via config')
        return

    command = server_config.get('command', 'uvx')
    python_version = server_config.get('python_version', '3.12')
    package = server_config.get('package', 'mcp-context-server[embeddings-ollama]<2.0.0')

    try:
        log_always('Pre-warming uvx cache...')
        # Use --help to trigger package download without starting server
        # Command list is built from trusted config values
        warmup_cmd = [str(command), '--python', str(python_version), '--with', str(package), '--help']
        result = subprocess.run(
            warmup_cmd,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            log_always('uvx cache pre-warm successful')
        else:
            log_always(f'uvx cache pre-warm returned code {result.returncode}', level='WARN')
    except subprocess.TimeoutExpired:
        log_always('uvx cache pre-warm timed out (continuing anyway)', level='WARN')
    except FileNotFoundError:
        log_always('uvx command not found (continuing anyway)', level='WARN')
    except Exception as e:
        # Silent failure for pre-warm
        log_always(f'uvx cache pre-warm failed: {type(e).__name__}: {e}', level='WARN')


def setup_windows_utf8() -> None:
    """
    Configure Windows console for UTF-8 encoding.

    This ensures that subprocess communication uses UTF-8 instead of
    Windows codepage (CP1252, Windows-1251) which corrupts non-ASCII text.

    CRITICAL for handling Cyrillic, Chinese, Arabic, and other non-ASCII text.
    Without this, non-ASCII characters stored via the hook appear as
    garbled text (mojibake) due to Windows default codepage encoding.

    This function:
    1. Sets PYTHONUTF8=1 environment variable (Python 3.7+)
    2. Configures Windows console codepage to UTF-8 (65001)

    It is applied before any subprocess operations to ensure proper
    encoding for FastMCP StdioTransport stdin/stdout communication.
    """
    log_always('Configuring Windows UTF-8 encoding')
    if sys.platform != 'win32':
        log_always('Not Windows platform, skipping UTF-8 setup')
        return

    try:
        # Run the subprocesses this hook starts, such as the stdio MCP server, in
        # Python's UTF-8 mode; this process's own streams keep their encoding,
        # which is why its payload is parsed from the stdin bytes
        os.environ['PYTHONUTF8'] = '1'

        # Set Windows console codepage to UTF-8 (65001)
        # This affects stdin/stdout/stderr of spawned subprocesses
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)  # Input codepage
        kernel32.SetConsoleOutputCP(65001)  # Output codepage

        log_always('UTF-8 mode configured for Windows console')
    except Exception as e:
        # Non-fatal: log error but continue
        # Hook should still work even if UTF-8 setup fails
        error_msg = f'Failed to set Windows UTF-8 mode: {e}'
        log_always(error_msg, level='ERROR')


def report_error(error_type: str, error_msg: str) -> None:
    """
    Report error to both debug log and stats file for better diagnostics.

    Creates an error tracking file at .claude/.hook_errors with structured
    error information for troubleshooting intermittent failures.

    Logging is CONDITIONAL based on CLAUDE_HOOK_DEBUG_ENABLED environment variable.

    Args:
        error_type: Category of error (e.g., 'UVX_FAILURE', 'THREAD_ID_READ')
        error_msg: Detailed error message
    """
    full_msg = f'{error_type}: {error_msg}'
    log_always(full_msg, level='ERROR')

    # Early exit if logging not enabled
    if not _LOGGING_ENABLED:
        return

    # Track error statistics
    project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
    if project_dir:
        stats_file = Path(project_dir) / '.claude' / '.hook_errors'
        try:
            stats = {
                'timestamp': datetime.now(tz=UTC).isoformat(),
                'error': error_type,
                'message': error_msg,
            }
            with stats_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
        except Exception:
            # Silent failure for stats tracking
            pass


class Deadline:
    """
    The run's one time budget, shared by every attempt the hook makes.

    Claude Code kills a hook at the timeout its environment registers and
    discards its output, so the hook keeps a shorter budget of its own and
    answers before that happens: every connection or request timeout is capped
    by what remains of the budget, and no attempt starts once less than
    MIN_ATTEMPT_SECONDS remains. Elapsed time is read from a monotonic clock
    against the start the budget was given, by default its creation.
    """

    MIN_ATTEMPT_SECONDS = 1.0

    def __init__(self, seconds: float, started_at: float | None = None) -> None:
        """
        Initialize the budget.

        Args:
            seconds: The whole budget, in seconds
            started_at: The time.monotonic() reading the budget counts from; now when omitted
        """
        self.seconds = seconds
        self.started_at = time.monotonic() if started_at is None else started_at

    def elapsed(self) -> float:
        """Return the seconds used since the start."""
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        """Return the seconds left, never below zero."""
        return max(0.0, self.seconds - self.elapsed())

    def cap(self, timeout: float) -> float:
        """Return the smaller of a timeout and the remaining budget."""
        return min(timeout, self.remaining())

    def allows_attempt(self) -> bool:
        """Return whether at least MIN_ATTEMPT_SECONDS of the budget remain for another attempt."""
        return self.remaining() >= self.MIN_ATTEMPT_SECONDS

    def describe(self) -> str:
        """Return the budget and its used part as a short phrase for log and error text."""
        return f'{self.elapsed():.1f}s of the {self.seconds:g}s hook deadline used'


# The image name of the process that launches the hook: Claude Code runs it as
# `uv run`, and uv prepares the script's environment before it starts the
# interpreter.
LAUNCHER_IMAGE = 'uv'

# The oldest launch still accepted as this run's, in seconds. Claude Code applies
# at most 600 seconds to a command hook by default, so a hook it is still waiting
# on was launched more recently than that; startups from a few seconds on a warm
# environment to more than a minute on a slow or memory-starved machine stay well
# inside it.
MAX_LAUNCH_AGE_SECONDS = 600.0

_PROC_ROOT = Path('/proc')

# struct proc_bsdinfo, which macOS proc_pidinfo(PROC_PIDTBSDINFO) fills: twelve
# 32-bit fields (pbi_ppid is the fifth), pbi_comm[16], pbi_name[32], six 32-bit
# fields, and the start time as 64-bit seconds and microseconds.
_BSDINFO = struct.Struct('<12I16s32s5Ii2Q')


@dataclass(frozen=True)
class ProcessRecord:
    """What the launch read needs to know about one process."""

    image: str  # image file name, without its directory
    path: str | None  # full image path, where the platform's reader reports one
    parent: int  # parent process id
    created: float  # creation time, in seconds on the platform's process clock


class ProcessTable(Protocol):
    """Reads process records and the clock their creation times are given on."""

    def read(self, pid: int) -> ProcessRecord | None:
        """Return the record of a process, or None when it cannot be read."""
        ...

    def now(self) -> float:
        """Return the current time on the clock creation times are given on."""
        ...


class _ProcessBasicInformation(ctypes.Structure):
    """PROCESS_BASIC_INFORMATION, as NtQueryInformationProcess fills it on Windows."""

    _fields_ = (
        ('ExitStatus', ctypes.c_long),
        ('PebBaseAddress', ctypes.c_void_p),
        ('AffinityMask', ctypes.c_size_t),
        ('BasePriority', ctypes.c_long),
        ('UniqueProcessId', ctypes.c_size_t),
        ('InheritedFromUniqueProcessId', ctypes.c_size_t),
    )


class WindowsProcessTable:
    """Process records read through the Win32 API; creation times are wall-clock seconds."""

    _QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_BASIC_INFORMATION = 0
    _MAX_PATH_CHARACTERS = 32768
    _FILETIME_TICKS_PER_SECOND = 10_000_000
    _FILETIME_TO_UNIX_SECONDS = 11_644_473_600  # from 1601-01-01 to 1970-01-01

    def read(self, pid: int) -> ProcessRecord | None:
        """Return a process's image, parent, and creation time, or None when it cannot be opened or read."""
        record: ProcessRecord | None = None
        if sys.platform == 'win32':
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel32.OpenProcess(self._QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
                times = (ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user))
                if not kernel32.GetProcessTimes(wintypes.HANDLE(handle), *times):
                    return None
                size = wintypes.DWORD(self._MAX_PATH_CHARACTERS)
                path = ctypes.create_unicode_buffer(self._MAX_PATH_CHARACTERS)
                if not kernel32.QueryFullProcessImageNameW(wintypes.HANDLE(handle), 0, path, ctypes.byref(size)):
                    return None
                info = _ProcessBasicInformation()
                status = ctypes.WinDLL('ntdll').NtQueryInformationProcess(
                    wintypes.HANDLE(handle), self._PROCESS_BASIC_INFORMATION, ctypes.byref(info), ctypes.sizeof(info), None,
                )
                if status != 0:
                    return None
            finally:
                kernel32.CloseHandle(handle)
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            record = ProcessRecord(
                image=ntpath.basename(path.value),
                path=path.value,
                parent=int(info.InheritedFromUniqueProcessId),
                created=ticks / self._FILETIME_TICKS_PER_SECOND - self._FILETIME_TO_UNIX_SECONDS,
            )
        return record

    def now(self) -> float:
        """Return the wall clock, the clock creation times are given on."""
        return time.time()


def parse_proc_stat(text: str) -> tuple[str, int, int] | None:
    """
    Parse a Linux /proc/<pid>/stat line into its comm, parent id, and start time.

    As proc(5) documents the line, the comm is field 2 in parentheses and may
    itself hold spaces and parentheses, so it ends at the last closing
    parenthesis; the parent id is field 4 and the start time after boot, in
    clock ticks, field 22.

    Returns:
        The comm, the parent process id, and the start time in clock ticks, or
        None for a line without those fields
    """
    opening = text.find('(')
    closing = text.rfind(')')
    if opening < 0 or closing < opening:
        return None
    fields = text[closing + 1:].split()
    try:
        return text[opening + 1:closing], int(fields[1]), int(fields[19])
    except (IndexError, ValueError):
        return None


def _linux_clock_ticks() -> int:
    """Return the clock ticks per second /proc start times are counted in."""
    if sys.platform != 'win32':
        return os.sysconf('SC_CLK_TCK')
    raise OSError('/proc clock ticks are not available on Windows')


def _linux_boottime() -> float:
    """Return the seconds since boot, counting suspended time as /proc start times do."""
    if sys.platform == 'linux':
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    raise OSError('CLOCK_BOOTTIME is available only on Linux')


class LinuxProcessTable:
    """Process records read from /proc; creation times are seconds since boot."""

    def __init__(self, proc_root: Path = _PROC_ROOT, clock_ticks: int | None = None) -> None:
        """
        Initialize the table.

        Args:
            proc_root: The procfs mount to read
            clock_ticks: Clock ticks per second of the start times; read from the system when omitted
        """
        self.proc_root = proc_root
        self.clock_ticks = clock_ticks if clock_ticks is not None else _linux_clock_ticks()

    def read(self, pid: int) -> ProcessRecord | None:
        """Return a process's comm, parent, and creation time, or None when its stat cannot be read."""
        try:
            text = (self.proc_root / str(pid) / 'stat').read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None
        fields = parse_proc_stat(text)
        if fields is None:
            return None
        comm, parent, start_ticks = fields
        return ProcessRecord(image=comm, path=None, parent=parent, created=start_ticks / self.clock_ticks)

    def now(self) -> float:
        """Return the boot clock, the clock creation times are given on."""
        return _linux_boottime()


def parse_proc_bsdinfo(buffer: bytes) -> tuple[str, int, float] | None:
    """
    Parse a macOS struct proc_bsdinfo into its comm, parent id, and start time.

    Returns:
        The comm, the parent process id, and the start time in wall-clock
        seconds, or None for a buffer shorter than the structure
    """
    if len(buffer) < _BSDINFO.size:
        return None
    fields = _BSDINFO.unpack_from(buffer)
    comm = fields[12].split(b'\0', 1)[0].decode('utf-8', 'replace')
    return comm, int(fields[4]), fields[20] + fields[21] / 1_000_000


class MacProcessTable:
    """Process records read through libproc; creation times are wall-clock seconds."""

    _PIDTBSDINFO = 3

    def read(self, pid: int) -> ProcessRecord | None:
        """Return a process's comm, parent, and creation time, or None when it cannot be read."""
        record: ProcessRecord | None = None
        if sys.platform == 'darwin':
            try:
                libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
            except OSError:
                return None
            libproc.proc_pidinfo.restype = ctypes.c_int
            libproc.proc_pidinfo.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
            buffer = ctypes.create_string_buffer(_BSDINFO.size)
            if libproc.proc_pidinfo(pid, self._PIDTBSDINFO, 0, buffer, _BSDINFO.size) != _BSDINFO.size:
                return None
            fields = parse_proc_bsdinfo(buffer.raw)
            if fields is None:
                return None
            comm, parent, created = fields
            record = ProcessRecord(image=comm, path=None, parent=parent, created=created)
        return record

    def now(self) -> float:
        """Return the wall clock, the clock creation times are given on."""
        return time.time()


def process_table() -> ProcessTable | None:
    """Return the process table of the running platform, or None where the hook has no reader."""
    table: ProcessTable | None = None
    if sys.platform == 'win32':
        table = WindowsProcessTable()
    elif sys.platform == 'linux':
        table = LinuxProcessTable()
    elif sys.platform == 'darwin':
        table = MacProcessTable()
    return table


def _is_launcher(record: ProcessRecord) -> bool:
    """Return whether a process is the uv launcher, by its image name in any case and with or without .exe."""
    return record.image.lower().removesuffix('.exe') == LAUNCHER_IMAGE


def _is_redirector(child: ProcessRecord, parent: ProcessRecord, executable: str) -> bool:
    """
    Return whether a parent is the venv redirector that started its child.

    A Windows venv runs its interpreter through a small python.exe in the
    environment that starts the base interpreter, and the base interpreter
    reports that redirector as sys.executable. So the parent is the redirector
    when its image is sys.executable and the child's is not.

    Returns:
        Whether the parent is the child's venv redirector
    """
    if parent.path is None or child.path is None:
        return False
    target = os.path.normcase(executable)
    return os.path.normcase(parent.path) == target and os.path.normcase(child.path) != target


def _parent_of(processes: ProcessTable, child: ProcessRecord) -> ProcessRecord | None:
    """
    Return the process that started a child.

    A parent created after its child holds a reused process id, so it is not the
    process that started the child.

    Returns:
        The parent's record, or None when it cannot be read or was created after the child
    """
    parent = processes.read(child.parent)
    if parent is None or parent.created > child.created:
        return None
    return parent


def _launcher_created(processes: ProcessTable, executable: str, pid: int) -> float | None:
    """
    Return the creation time of the uv process that launched the hook.

    The launcher is the hook's parent or, when that parent is the venv
    redirector of the hook's interpreter, the redirector's parent; no other
    process is taken for it.

    Returns:
        The launcher's creation time on the table's clock, or None when the
        launcher cannot be identified
    """
    hook = processes.read(pid)
    if hook is None:
        return None
    parent = _parent_of(processes, hook)
    if parent is not None and _is_redirector(hook, parent, executable):
        parent = _parent_of(processes, parent)
    if parent is None or not _is_launcher(parent):
        return None
    return parent.created


def launcher_clock_start(
    hook_started_at: float,
    processes: ProcessTable | None,
    executable: str,
    pid: int,
    monotonic: Callable[[], float] = time.monotonic,
) -> float:
    """
    Return the time.monotonic() reading the run's budget counts from.

    The budget counts from the creation of the uv process that launched the
    hook, converted once to the monotonic clock by taking its age, read on the
    process table's clock, off a monotonic reading taken beside it. The hook's
    own clock is used instead when the launcher cannot be identified, when its
    age is above MAX_LAUNCH_AGE_SECONDS, or when the result would fall after
    the hook's own clock, since the launch precedes the module code; that last
    case includes a creation time apparently later than the read.

    Args:
        hook_started_at: The hook's own clock, read when its module code started
        processes: The platform's process table, or None where there is none
        executable: The interpreter's sys.executable
        pid: The hook's process id
        monotonic: The monotonic clock

    Returns:
        The monotonic reading of the launch, or hook_started_at
    """
    if processes is None:
        return hook_started_at
    created = _launcher_created(processes, executable, pid)
    if created is None:
        return hook_started_at
    age = processes.now() - created
    started_at = monotonic() - age
    if age > MAX_LAUNCH_AGE_SECONDS or started_at > hook_started_at:
        return hook_started_at
    return started_at


@dataclass(frozen=True)
class StartTimes:
    """When the hook's interpreter, and the venv redirector that started it, were created, in wall-clock seconds."""

    interpreter: float | None  # None when the hook's own process cannot be read
    redirector: float | None  # None unless the interpreter's parent is its venv redirector, which only Windows runs


def start_times(processes: ProcessTable | None, executable: str, pid: int) -> StartTimes:
    """
    Return when the hook's interpreter, and the venv redirector that started it, were created.

    Each creation time is read on the process table's clock and converted by
    taking its age off the wall clock. The redirector is timed only when the
    interpreter's parent is the venv redirector of that interpreter; any other
    parent, an unreadable one, or one newer than the interpreter is not.

    Args:
        processes: The platform's process table, or None where there is none
        executable: The interpreter's sys.executable
        pid: The hook's process id

    Returns:
        The creation times in wall-clock seconds, each None when it cannot be read
    """
    if processes is None:
        return StartTimes(interpreter=None, redirector=None)
    hook = processes.read(pid)
    if hook is None:
        return StartTimes(interpreter=None, redirector=None)
    parent = _parent_of(processes, hook)
    redirector = parent if parent is not None and _is_redirector(hook, parent, executable) else None
    wall_offset = time.time() - processes.now()
    return StartTimes(
        interpreter=hook.created + wall_offset,
        redirector=None if redirector is None else redirector.created + wall_offset,
    )


def own_start_times() -> StartTimes:
    """Return when the running hook's interpreter, and the venv redirector that started it, were created."""
    return start_times(process_table(), sys.executable, os.getpid())


# The directory under the temporary directory that holds each run's startup trace
STARTUP_TRACE_DIRECTORY = 'claude_hook_user_prompt_context_saver_traces'

# The environment variable that turns the startup trace off when set to 0, false, or no
STARTUP_TRACE_VARIABLE = 'CLAUDE_HOOK_STARTUP_TRACE'

# The most traces a run leaves in the trace directory, its own included. A trace
# that outlives its run belongs to a run that was killed or crashed, so only the
# newest of those are kept.
MAX_KEPT_TRACES = 20


def startup_trace_directory(environ: Mapping[str, str] = os.environ) -> Path | None:
    """
    Return the directory a run's startup trace goes to, or None when the trace is off.

    Args:
        environ: The environment to read CLAUDE_HOOK_STARTUP_TRACE from

    Returns:
        The trace directory under the temporary directory, or None when the
        variable is 0, false, or no, in any case
    """
    if environ.get(STARTUP_TRACE_VARIABLE, '').strip().lower() in ('0', 'false', 'no'):
        return None
    return Path(tempfile.gettempdir()) / STARTUP_TRACE_DIRECTORY


def _utc_text(wall: float) -> str:
    """Return wall-clock seconds as ISO 8601 text in UTC, to the millisecond."""
    return datetime.fromtimestamp(wall, tz=UTC).isoformat(timespec='milliseconds')


def prune_startup_traces(directory: Path, keep: int, min_age_seconds: float) -> None:
    """
    Remove the traces beyond the newest few, sparing any that may belong to a live run.

    A trace's name starts with its run's start time, so the names sort oldest
    first. A trace written within the last min_age_seconds may belong to a run
    still in flight, so it is never removed, whatever its place in that order.
    Files that are not traces are left alone, and a file that cannot be read or
    removed is skipped.

    Args:
        directory: The trace directory
        keep: How many of the newest traces always stay
        min_age_seconds: The least time since a trace was last written before it may be removed
    """
    now = time.time()
    try:
        with os.scandir(directory) as entries:
            traces = sorted(
                (entry for entry in entries if entry.name.endswith('.jsonl') and entry.is_file()),
                key=lambda entry: entry.name,
                reverse=True,
            )
    except OSError:
        return
    for entry in traces[keep:]:
        with suppress(OSError):
            if now - entry.stat().st_mtime > min_age_seconds:
                os.remove(entry.path)


class StartupTrace:
    """
    The stages one run reaches, kept on disk for as long as the run lasts.

    Claude Code kills a hook at its registered timeout and discards its output,
    so a run that stalls says nothing about where. The trace appends one JSON
    line per stage to a file of its own, flushed as it is written so that it
    survives the process being killed, and finish() removes the file when the
    run ends normally. Every record names its stage, the wall-clock time the run
    reached it (at, ISO 8601 in UTC), and the seconds since the hook's module
    code started (elapsed); some records add fields of their own. A trace whose
    file cannot be created records nothing, and one whose writes fail stops
    recording; neither breaks the hook.
    """

    def __init__(self, path: Path | None, stream: TextIO | None, started_wall: float, started_monotonic: float) -> None:
        """
        Initialize the trace; begin() is the way to start one.

        Args:
            path: The trace file, or None for a trace that records nothing
            stream: The open trace file, or None for a trace that records nothing
            started_wall: The wall-clock time the hook's module code started
            started_monotonic: The time.monotonic() reading taken beside it
        """
        self.path = path
        self._stream = stream
        self._started_wall = started_wall
        self._started_monotonic = started_monotonic

    @classmethod
    def begin(
        cls, directory: Path | None, started_wall: float, started_monotonic: float, pid: int | None = None,
        **fields: object,
    ) -> 'StartupTrace':
        """
        Create the run's trace file, write its module_start record, and trim the traces of earlier runs.

        The file is named for the start time in UTC and the process id, so the
        newest trace sorts last and names the process that left it.

        Args:
            directory: The trace directory, or None when the trace is off
            started_wall: The wall-clock time the hook's module code started
            started_monotonic: The time.monotonic() reading taken beside it
            pid: The run's process id; this process's when omitted
            **fields: More fields for the module_start record

        Returns:
            The run's trace, recording nothing when the directory is None or the file cannot be created
        """
        if directory is None:
            return cls(None, None, started_wall, started_monotonic)
        run_pid = os.getpid() if pid is None else pid
        started = datetime.fromtimestamp(started_wall, tz=UTC)
        path = directory / f'{started:%Y%m%dT%H%M%S}.{started.microsecond // 1000:03d}Z-{run_pid}.jsonl'
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stream = path.open('x', encoding='utf-8')
        except OSError:
            return cls(None, None, started_wall, started_monotonic)
        trace = cls(path, stream, started_wall, started_monotonic)
        trace._write({'stage': 'module_start', 'at': _utc_text(started_wall), 'elapsed': 0.0, 'pid': run_pid, **fields})
        prune_startup_traces(directory, MAX_KEPT_TRACES, MAX_LAUNCH_AGE_SECONDS)
        return trace

    @property
    def recording(self) -> bool:
        """Whether a record written now reaches the trace file: the file was created and no write has failed."""
        return self._stream is not None

    def record(self, stage: str, **fields: object) -> None:
        """
        Append the record of a stage the run has just reached.

        Args:
            stage: The stage's name
            **fields: More fields for the record
        """
        elapsed = round(time.monotonic() - self._started_monotonic, 3)
        self._write({'stage': stage, 'at': _utc_text(time.time()), 'elapsed': elapsed, **fields})

    def wall_text(self, monotonic_reading: float) -> str:
        """
        Return a time.monotonic() reading of this run as ISO 8601 wall-clock text in UTC.

        Args:
            monotonic_reading: The reading to convert

        Returns:
            The wall-clock time of the reading
        """
        return _utc_text(self._started_wall + (monotonic_reading - self._started_monotonic))

    def finish(self) -> None:
        """Close and remove the trace file: the run ended normally."""
        self._close()
        if self.path is not None:
            with suppress(OSError):
                self.path.unlink()

    def _write(self, record: dict[str, object]) -> None:
        """Write one record and flush it to the file, closing the file for good when the write fails."""
        if self._stream is None:
            return
        try:
            self._stream.write(json.dumps(record) + '\n')
            self._stream.flush()
        except OSError:
            self._close()

    def _close(self) -> None:
        """Close the trace file once; the trace records nothing after it."""
        if self._stream is not None:
            with suppress(OSError):
                self._stream.close()
            self._stream = None


# The trace main() records into when it runs without one, such as when it is called directly
DISABLED_TRACE = StartupTrace(None, None, _HOOK_STARTED_WALL, _HOOK_STARTED_AT)


class UnconfirmedStoreError(Exception):
    """
    A run that ended without the server confirming the message stored.

    Attributes:
        cause: The last attempt's own failure, or a TimeoutError naming the budget
            when no attempt could start within it
        request_sent: Whether a store request had been sent when the run ended; when
            it had, the server may still complete it
        stored_context_ids: The context_ids of the chunks the server confirmed before
            the run ended (chunked store only)
        total_chunks: The number of chunks the message was split into, or None for a
            single store
    """

    def __init__(
        self,
        cause: Exception,
        request_sent: bool,
        stored_context_ids: list[int] | None = None,
        total_chunks: int | None = None,
    ) -> None:
        """
        Initialize the error.

        Args:
            cause: The last attempt's own failure
            request_sent: Whether a store request had been sent when the run ended
            stored_context_ids: The context_ids of the chunks the server confirmed
            total_chunks: The number of chunks, or None for a single store
        """
        super().__init__(f'{type(cause).__name__}: {cause}')
        self.cause = cause
        self.request_sent = request_sent
        self.stored_context_ids = list(stored_context_ids or [])
        self.total_chunks = total_chunks


def _unconfirmed_store_error(
    deadline: Deadline,
    last_error: Exception | None,
    request_sent: bool,
    max_attempts: int,
    error_type: str,
    stored_context_ids: list[int] | None = None,
    total_chunks: int | None = None,
) -> UnconfirmedStoreError:
    """
    Build and record the error that ends a run whose message was not confirmed stored.

    Args:
        deadline: The run's budget
        last_error: The failure of the last attempt, or None when no attempt could start
        request_sent: Whether the last attempt had sent its store request when it failed
        max_attempts: The configured number of connection attempts
        error_type: The category the failure is reported under
        stored_context_ids: The context_ids of the chunks the server confirmed (chunked store only)
        total_chunks: The number of chunks, or None for a single store

    Returns:
        The error for the caller to raise, carrying the last attempt's own error (or
        a TimeoutError naming the budget when no attempt could start within it), the
        stage the run reached, and the chunks already stored.
    """
    cause: Exception = (
        last_error if last_error is not None
        else TimeoutError(f'no connection attempt could start ({deadline.describe()})')
    )
    error = UnconfirmedStoreError(cause, request_sent, stored_context_ids, total_chunks)
    if request_sent:
        detail = f'store request sent but not confirmed ({deadline.describe()}): {error}'
        if total_chunks is not None:
            detail += f'; chunks confirmed: {len(error.stored_context_ids)}/{total_chunks} {error.stored_context_ids}'
    else:
        detail = f'no store request was sent within {max_attempts} connection attempt(s) ({deadline.describe()}): {error}'
    log_always(detail, level='ERROR')
    report_error(error_type, detail)
    return error


class SyncMCPClient:
    """
    Synchronous wrapper for the async FastMCP client.

    This wrapper allows us to use the async FastMCP client in a synchronous
    context, which is required for Claude Code hooks. Every attempt it makes
    runs under the run's time budget, and a store request is sent at most once.
    """

    def __init__(
        self,
        server_command: list[str] | str,
        timeout: float = 120.0,
        max_connection_retries: int = 3,
        connection_timeout: float = 30.0,
        deadline: Deadline | None = None,
    ) -> None:
        """
        Initialize the synchronous MCP client wrapper.

        Args:
            server_command: Command to start the MCP server (list or string)
            timeout: Timeout in seconds per attempt, capped by the run's remaining budget
            max_connection_retries: Number of connection attempts before the store request is sent
            connection_timeout: Timeout in seconds per connection attempt
            deadline: The run's time budget; the default budget, starting now, when omitted
        """
        self.server_command = server_command
        self.timeout = timeout
        self.max_connection_retries = max_connection_retries
        self.connection_timeout = connection_timeout
        self.deadline = deadline if deadline is not None else Deadline(DEFAULT_CONFIG['mcp_client']['deadline_seconds'])
        log_always(
            f'SyncMCPClient initialized: command={server_command}, timeout={timeout}, '
            f'max_conn_retries={max_connection_retries}, conn_timeout={connection_timeout}, '
            f'deadline={self.deadline.seconds}s',
        )

    def _run_async(self, coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
        """
        Run an async coroutine in a sync context.

        Args:
            coro: The async coroutine to run

        Returns:
            The result of the coroutine

        Raises:
            RuntimeError: If called from an existing async context
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            # If we get here, we're already in an async context
            raise RuntimeError('Cannot run from async context')
        except RuntimeError as e:
            if 'no running event loop' in str(e).lower():
                # No event loop, safe to create one
                return asyncio.run(coro)
            # Re-raise if it's the "already in async" error
            raise

    def _calculate_message_size(self, thread_id: str, source: str, text: str) -> int:
        """
        Calculate the JSON message size in bytes.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context
            text: The text content to store

        Returns:
            Size of the JSON message in bytes
        """
        test_json = json.dumps({'thread_id': thread_id, 'source': source, 'text': text})
        return len(test_json.encode('utf-8'))

    @staticmethod
    def _chunk_text_by_bytes(text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks based on byte size, respecting UTF-8 boundaries.

        This method ensures that:
        1. Each chunk is at most `chunk_size` bytes when UTF-8 encoded
        2. No multi-byte UTF-8 characters are split mid-sequence
        3. All chunks decode back to valid UTF-8 strings

        UTF-8 encoding rules used for boundary detection:
        - 1-byte chars (ASCII): 0xxxxxxx (0x00-0x7F)
        - 2-byte chars: 110xxxxx 10xxxxxx
        - 3-byte chars: 1110xxxx 10xxxxxx 10xxxxxx
        - 4-byte chars: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        - Continuation bytes: 10xxxxxx (0x80-0xBF)

        Args:
            text: The text to split into chunks
            chunk_size: Maximum size in bytes for each chunk

        Returns:
            List of text chunks, each at most `chunk_size` bytes when encoded
        """
        if not text:
            return []

        text_bytes = text.encode('utf-8')
        chunks: list[str] = []

        i = 0
        while i < len(text_bytes):
            # Take up to chunk_size bytes
            end = min(i + chunk_size, len(text_bytes))
            chunk_bytes = text_bytes[i:end]

            # If we're not at the end, ensure we don't split a multi-byte character
            if end < len(text_bytes):
                # Walk back if we're in the middle of a UTF-8 sequence
                # Continuation bytes have pattern 10xxxxxx (0x80-0xBF)
                while chunk_bytes and (chunk_bytes[-1] & 0xC0) == 0x80:
                    chunk_bytes = chunk_bytes[:-1]

                # Safety check: if we walked back to nothing, something is wrong
                # This should never happen with valid UTF-8, but handle gracefully
                if not chunk_bytes:
                    log_always(
                        f'UTF-8 boundary detection failed at position {i}, '
                        f'forcing single character chunk',
                        level='WARN',
                    )
                    # Find the start byte and include the full character
                    start_byte = text_bytes[i]
                    if (start_byte & 0x80) == 0x00:
                        char_len = 1  # ASCII
                    elif (start_byte & 0xE0) == 0xC0:
                        char_len = 2  # 2-byte char
                    elif (start_byte & 0xF0) == 0xE0:
                        char_len = 3  # 3-byte char
                    elif (start_byte & 0xF8) == 0xF0:
                        char_len = 4  # 4-byte char
                    else:
                        char_len = 1  # Fallback for malformed UTF-8
                    chunk_bytes = text_bytes[i : i + char_len]

            # Decode the chunk and add to list
            try:
                chunk_str = chunk_bytes.decode('utf-8')
                chunks.append(chunk_str)
            except UnicodeDecodeError as e:
                # This should never happen if our boundary logic is correct
                log_always(
                    f'UTF-8 decode error at position {i}: {e}. '
                    f'Chunk bytes (first 20): {chunk_bytes[:20]!r}',
                    level='ERROR',
                )
                # Skip problematic bytes and continue
                i += 1
                continue

            i += len(chunk_bytes)

        return chunks

    @staticmethod
    def _response_payload(result: object) -> dict[str, Any]:
        """
        Return the dictionary a store_context response carries.

        FastMCP delivers the tool's structured output as structured_content; a
        response without one is returned as it is.

        Args:
            result: The value call_tool returned

        Returns:
            The structured content when present, otherwise the result itself
        """
        structured_content = getattr(result, 'structured_content', None)
        if structured_content is not None:
            log_always('Using structured_content for stdio response')
            return cast(dict[str, Any], structured_content)
        return cast(dict[str, Any], result)

    async def _store_single_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a single context message asynchronously using the MCP server.

        Each attempt starts the server, connects, and sends one store_context
        request, under a timeout capped by the run's remaining budget. A failure
        before the request is sent (the server did not start, the connection was
        not established) is retried with exponential backoff while the budget
        allows another attempt. A failure or timeout after the request was sent
        ends the run without a retry: the server may have received the request,
        and a second copy would duplicate the entry. A run that ends without a
        confirmed store raises UnconfirmedStoreError, carrying the last attempt's
        own error and whether the request was sent.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        import asyncio

        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        log_always(f'_store_single_context_async called: thread_id={thread_id}, source={source}, text_len={len(text)}')

        if isinstance(self.server_command, list):
            cmd = self.server_command[0]
            args = self.server_command[1:] if len(self.server_command) > 1 else []
        else:
            parts = self.server_command.split()
            cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []

        log_always(f'MCP server command: {cmd} {args}')

        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'

        last_error: Exception | None = None
        max_retries = self.max_connection_retries
        request_sent = False

        for attempt in range(max_retries):
            if not self.deadline.allows_attempt():
                log_always(f'No further connection attempt: {self.deadline.describe()}', level='WARN')
                break
            attempt_timeout = self.deadline.cap(self.timeout)
            try:
                log_always(f'Connection attempt {attempt + 1}/{max_retries} (timeout={attempt_timeout}s)')

                transport = cast(Any, StdioTransport(cmd, args, env=env))
                log_always('StdioTransport created')

                # The timeout covers the connection and the request together
                async with asyncio.timeout(attempt_timeout):
                    async with cast(Any, Client(transport)) as client:
                        # Connection health check - log successful connection
                        log_always('MCP client connected successfully (health check passed)')

                        # Normalize Windows line endings for NDJSON format
                        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

                        # Build call parameters
                        params: dict[str, Any] = {
                            'thread_id': thread_id,
                            'source': source,
                            'text': normalized_text,
                        }
                        if metadata:
                            params['metadata'] = metadata

                        log_always(f'Calling store_context with thread_id={thread_id}, source={source}')
                        # From here on the request may have reached the server, so a
                        # failure ends the run instead of sending a second copy
                        request_sent = True
                        result = await client.call_tool('store_context', params)
                        log_always(f'store_context returned: {type(result).__name__}')
                        return self._response_payload(result)

            except TimeoutError:
                stage = 'store_context request' if request_sent else 'connection attempt'
                last_error = TimeoutError(f'{stage} timed out after {attempt_timeout}s')
                log_always(f'Connection attempt {attempt + 1}/{max_retries}: {last_error}', level='WARN')
            except Exception as e:
                last_error = e
                log_always(
                    f'Connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}',
                    level='WARN',
                )

            if request_sent:
                break

            # Exponential backoff before retry (1s, 2s, 4s), within the remaining budget
            if attempt < max_retries - 1:
                backoff_delay = min(1.0 * (2**attempt), self.deadline.remaining())
                log_always(f'Waiting {backoff_delay}s before retry (exponential backoff)')
                await asyncio.sleep(backoff_delay)

        failure = _unconfirmed_store_error(self.deadline, last_error, request_sent, max_retries, 'MCP_CONNECTION_EXHAUSTED')
        raise failure from failure.cause

    async def _store_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context asynchronously using the MCP server with automatic chunking.

        This method handles large messages by splitting them into chunks when they
        exceed the Windows subprocess pipe buffer limit (~64KB). Messages are split
        at safe boundaries to avoid breaking UTF-8 encoding.

        One MCP connection carries all chunks, each sent exactly once; a chunk
        whose request fails after it was sent is reported as failed rather than
        sent again.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        # Calculate message size to check if chunking is needed
        message_size = self._calculate_message_size(thread_id, source, text)
        log_always(f'Message size: {len(text)} chars, {message_size} bytes (limit: {_MAX_MESSAGE_SIZE} bytes)')

        # If message is small enough, send directly
        if message_size <= _MAX_MESSAGE_SIZE:
            log_always('Message size within limits, sending directly')
            return await self._store_single_context_async(thread_id, source, text, metadata)

        # Message is too large, need to chunk it
        log_always(
            f'Message too large ({message_size} bytes > {_MAX_MESSAGE_SIZE} bytes), chunking required',
            level='WARN',
        )

        # Use byte-based chunking with UTF-8 boundary handling
        # This ensures each chunk respects the byte limit even for multi-byte characters
        chunks = self._chunk_text_by_bytes(text, _CHUNK_SIZE)

        # Log chunk statistics for debugging
        text_bytes = len(text.encode('utf-8'))
        chunk_byte_sizes = [len(c.encode('utf-8')) for c in chunks]
        log_always(
            f'Split {text_bytes} bytes into {len(chunks)} chunks: '
            f'sizes={chunk_byte_sizes}, avg={sum(chunk_byte_sizes) / len(chunks):.0f} bytes',
        )

        # One connection carries every chunk
        return await self._store_chunks_single_connection(
            thread_id=thread_id,
            source=source,
            chunks=chunks,
            text_bytes=text_bytes,
        )

    async def _store_chunks_single_connection(
        self,
        thread_id: str,
        source: str,
        chunks: list[str],
        text_bytes: int,
    ) -> dict[str, Any]:
        """
        Store all chunks through one MCP connection.

        One connection carries every chunk, each sent as its own store_context
        request exactly once: a chunk whose request fails after it was sent is
        recorded as failed rather than sent again, because the server may have
        received it, and the remaining chunks still go out. A failure before the
        first request is sent is retried with exponential backoff while the
        run's budget allows another attempt, and the whole exchange runs under a
        timeout capped by that budget. An exchange that does not finish raises
        UnconfirmedStoreError, carrying the last connection attempt's own error,
        whether a chunk was sent, and the context_ids of the chunks the server
        confirmed before it ended.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context
            chunks: List of text chunks to store
            text_bytes: Total size in bytes for logging

        Returns:
            Summary of chunked storage results
        """
        import asyncio

        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        log_always(f'_store_chunks_single_connection: storing {len(chunks)} chunks through one connection')

        if isinstance(self.server_command, list):
            cmd = self.server_command[0]
            args = self.server_command[1:] if len(self.server_command) > 1 else []
        else:
            parts = self.server_command.split()
            cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []

        log_always(f'MCP server command: {cmd} {args}')

        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'

        last_error: Exception | None = None
        max_retries = self.max_connection_retries
        request_sent = False
        # Only the attempt that sends chunks fills this, and no attempt follows it
        results: list[dict[str, Any]] = []

        for conn_attempt in range(max_retries):
            if not self.deadline.allows_attempt():
                log_always(f'No further connection attempt: {self.deadline.describe()}', level='WARN')
                break
            # Extended timeout for chunked storage: base timeout * number of chunks, within the budget
            extended_timeout = self.deadline.cap(self.timeout * max(len(chunks), 2))
            try:
                log_always(f'Connection attempt {conn_attempt + 1}/{max_retries} (timeout={extended_timeout}s)')

                transport = cast(Any, StdioTransport(cmd, args, env=env))
                log_always('StdioTransport created for chunked storage')

                # The timeout covers the connection and every chunk together
                async with asyncio.timeout(extended_timeout):
                    async with cast(Any, Client(transport)) as client:
                        log_always('MCP client connected for chunked storage (one connection for all chunks)')

                        for idx, chunk in enumerate(chunks):
                            chunk_num = idx + 1
                            chunk_byte_size = len(chunk.encode('utf-8'))
                            log_always(
                                f'Storing chunk {chunk_num}/{len(chunks)} '
                                f'({len(chunk)} chars, {chunk_byte_size} bytes)',
                            )

                            # Add metadata to track chunks. Chunk entries are user
                            # messages too, so they carry the same universal schema
                            # v1 discriminator as the single-store path.
                            metadata = {
                                'schema_version': 1,
                                'kind': 'user_message',
                                'chunk': chunk_num,
                                'total_chunks': len(chunks),
                                'chunk_size_chars': len(chunk),
                                'chunk_size_bytes': chunk_byte_size,
                                'is_chunked': True,
                            }

                            # Normalize Windows line endings for NDJSON format
                            normalized_chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')

                            params: dict[str, Any] = {
                                'thread_id': thread_id,
                                'source': source,
                                'text': normalized_chunk,
                                'metadata': metadata,
                            }

                            # From here on the request may have reached the server, so a
                            # failed chunk is recorded instead of being sent again
                            request_sent = True
                            try:
                                result = await client.call_tool('store_context', params)
                                results.append(self._response_payload(result))
                                log_always(f'Chunk {chunk_num}/{len(chunks)} stored successfully')
                            except Exception as chunk_error:
                                log_always(
                                    f'Chunk {chunk_num}/{len(chunks)} failed: {type(chunk_error).__name__}: {chunk_error}',
                                    level='ERROR',
                                )
                                results.append({'error': str(chunk_error), 'chunk': chunk_num})

                        # All chunks processed - return results
                        successful_chunks = [r for r in results if 'error' not in r]
                        failed_chunks = [r for r in results if 'error' in r]

                        log_always(
                            f'Chunked storage complete: {len(successful_chunks)}/{len(chunks)} succeeded, '
                            f'{len(failed_chunks)} failed',
                        )

                        return {
                            'success': len(failed_chunks) == 0,
                            'chunked': True,
                            'total_chunks': len(chunks),
                            'chunks_stored': len(successful_chunks),
                            'chunks_failed': len(failed_chunks),
                            'total_bytes': text_bytes,
                            'results': results,
                            'failed_chunk_numbers': [r['chunk'] for r in failed_chunks],
                        }

            except TimeoutError:
                stage = 'chunked store_context exchange' if request_sent else 'connection attempt'
                last_error = TimeoutError(f'{stage} timed out after {extended_timeout}s')
                log_always(f'Connection attempt {conn_attempt + 1}/{max_retries}: {last_error}', level='WARN')
            except Exception as e:
                last_error = e
                log_always(
                    f'Connection attempt {conn_attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}',
                    level='WARN',
                )

            if request_sent:
                break

            # Exponential backoff before retry (1s, 2s, 4s), within the remaining budget
            if conn_attempt < max_retries - 1:
                backoff_delay = min(1.0 * (2**conn_attempt), self.deadline.remaining())
                log_always(f'Waiting {backoff_delay}s before connection retry (exponential backoff)')
                await asyncio.sleep(backoff_delay)

        stored_context_ids = [
            r['context_id'] for r in results if 'error' not in r and r.get('context_id') is not None
        ]
        failure = _unconfirmed_store_error(
            self.deadline, last_error, request_sent, max_retries, 'MCP_CHUNKED_CONNECTION_EXHAUSTED',
            stored_context_ids=stored_context_ids, total_chunks=len(chunks),
        )
        raise failure from failure.cause

    def store_context(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context synchronously.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        log_always(f'store_context called: thread_id={thread_id}, source={source}, text_len={len(text)}')
        result = self._run_async(self._store_context_async(thread_id, source, text, metadata))
        log_always('store_context completed successfully')
        return result


class FastMCPHttpClient:
    """
    HTTP-based MCP client using FastMCP's StreamableHttpTransport.

    This client uses the official FastMCP Client library for HTTP communication,
    providing robust protocol handling, automatic retries, and proper error management.

    The FastMCP Client is async-only, so this class wraps operations with asyncio.run()
    for synchronous hook execution.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        max_retries: int = 3,
        config: dict[str, Any] | None = None,
        deadline: Deadline | None = None,
    ) -> None:
        """
        Initialize the FastMCP HTTP client.

        Args:
            url: The MCP server endpoint URL
            timeout: Request timeout in seconds per attempt, capped by the run's remaining budget
            headers: Optional custom headers for authentication
            max_retries: Number of connection attempts before the store request is sent
            config: Optional configuration dictionary
            deadline: The run's time budget; the default budget, starting now, when omitted
        """
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}
        self.max_retries = max_retries
        self.config = config or DEFAULT_CONFIG
        self.deadline = deadline if deadline is not None else Deadline(DEFAULT_CONFIG['mcp_client']['deadline_seconds'])
        log_always(
            f'FastMCPHttpClient initialized: url={url}, timeout={timeout}, '
            f'max_retries={max_retries}, deadline={self.deadline.seconds}s',
        )

    def _calculate_message_size(self, thread_id: str, source: str, text: str) -> int:
        """Calculate the JSON message size in bytes."""
        test_json = json.dumps({'thread_id': thread_id, 'source': source, 'text': text})
        return len(test_json.encode('utf-8'))

    async def _store_context_async(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context asynchronously using FastMCP Client.

        Each attempt connects and sends one store_context request, under a
        timeout capped by the run's remaining budget. A failure before the
        request is sent (the connection or the MCP handshake failed) is retried
        with exponential backoff while the budget allows another attempt. A
        failure, a tool error, or a timeout after the request was sent ends the
        run without a retry: the server may have received the request, and a
        second copy would duplicate the entry. A run that ends without a
        confirmed store raises UnconfirmedStoreError, carrying the last attempt's
        own error and whether the request was sent.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        import asyncio

        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        log_always(f'_store_context_async: thread_id={thread_id}, source={source}, text_len={len(text)}')

        # Normalize Windows line endings
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

        last_error: Exception | None = None
        request_sent = False

        for attempt in range(self.max_retries):
            if not self.deadline.allows_attempt():
                log_always(f'No further connection attempt: {self.deadline.describe()}', level='WARN')
                break
            attempt_timeout = self.deadline.cap(self.timeout)
            try:
                log_always(f'FastMCP HTTP attempt {attempt + 1}/{self.max_retries} (timeout={attempt_timeout}s)')

                # Create transport with custom headers
                transport = StreamableHttpTransport(
                    url=self.url,
                    headers=self.headers or None,
                )

                # The timeout covers the connection and the request together
                async with asyncio.timeout(attempt_timeout):
                    async with Client(transport, timeout=attempt_timeout) as client:
                        log_always('FastMCP client connected successfully')

                        # Build call parameters
                        call_params: dict[str, Any] = {
                            'thread_id': thread_id,
                            'source': source,
                            'text': normalized_text,
                        }
                        if metadata:
                            call_params['metadata'] = metadata

                        # From here on the request may have reached the server, so a
                        # failure ends the run instead of sending a second copy
                        request_sent = True
                        result = await client.call_tool(
                            'store_context',
                            call_params,
                            timeout=attempt_timeout,
                            raise_on_error=True,
                        )

                        log_always(f'FastMCP call_tool returned: {type(result).__name__}')

                        # PRIORITY 1: Use structured_content (canonical FastMCP approach)
                        # FastMCP docs: "Raw structured JSON is also available via result.structured_content"
                        if hasattr(result, 'structured_content') and result.structured_content is not None:
                            log_always(
                                f'Using structured_content (dict) for response: '
                                f'{type(result.structured_content).__name__}',
                            )
                            # Explicit type annotation to satisfy both mypy and pyright
                            structured_dict: dict[str, Any] = result.structured_content
                            return structured_dict

                        # PRIORITY 2: Handle result.data with comprehensive fallback
                        if hasattr(result, 'data') and result.data is not None:
                            data = result.data

                            # Pydantic v2
                            if hasattr(data, 'model_dump') and callable(data.model_dump):
                                log_always(f'Using model_dump() for: {type(data).__name__}')
                                return cast(dict[str, Any], data.model_dump())

                            # Pydantic v1
                            if hasattr(data, 'dict') and callable(data.dict):
                                log_always(f'Using dict() for: {type(data).__name__}')
                                return cast(dict[str, Any], data.dict())

                            # RootModel with .root attribute
                            if hasattr(data, 'root'):
                                root_data = data.root
                                if isinstance(root_data, dict):
                                    log_always(f'Using .root for: {type(data).__name__}')
                                    return cast(dict[str, Any], root_data)

                            # Direct dict
                            if isinstance(data, dict):
                                log_always('Using direct dict')
                                return cast(dict[str, Any], data)

                            # Final fallback - log as INFO not WARN (this path is acceptable)
                            log_always(f'Using string fallback for: {type(data).__name__}')
                            return {'success': True, 'raw_data': str(data)}

                        # PRIORITY 3: Fallback to content block parsing
                        if hasattr(result, 'content') and result.content:
                            first_content = result.content[0]
                            content_text = getattr(first_content, 'text', None)
                            if content_text is not None:
                                try:
                                    log_always('Using content block parsing')
                                    return cast(dict[str, Any], json.loads(str(content_text)))
                                except json.JSONDecodeError:
                                    return {'success': True, 'raw_content': str(content_text)}

                        # No usable data
                        log_always('[WARN] No structured_content, data, or content available')
                        return {'success': False, 'error': 'No response data available'}

            except TimeoutError:
                stage = 'store_context request' if request_sent else 'connection attempt'
                last_error = TimeoutError(f'{stage} timed out after {attempt_timeout}s')
                log_always(f'FastMCP attempt {attempt + 1}/{self.max_retries}: {last_error}', level='WARN')
            except Exception as e:
                last_error = e
                log_always(
                    f'FastMCP attempt {attempt + 1}/{self.max_retries} failed: '
                    f'{type(e).__name__}: {e}',
                    level='WARN',
                )

            if request_sent:
                break

            # Exponential backoff before retry (1s, 2s, 4s), within the remaining budget
            if attempt < self.max_retries - 1:
                backoff_delay = min(1.0 * (2 ** attempt), self.deadline.remaining())
                log_always(f'Waiting {backoff_delay}s before retry')
                await asyncio.sleep(backoff_delay)

        failure = _unconfirmed_store_error(
            self.deadline, last_error, request_sent, self.max_retries, 'FASTMCP_HTTP_CONNECTION_EXHAUSTED',
        )
        raise failure from failure.cause

    def store_context(
        self,
        thread_id: str,
        source: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store context via FastMCP HTTP client synchronously.

        This method wraps the async FastMCP client with asyncio.run() for
        synchronous hook execution.

        Args:
            thread_id: The thread/session identifier
            source: The source of the context (always "user" for this hook)
            text: The prompt text to store
            metadata: Optional metadata to include with the context

        Returns:
            The server response as a dictionary
        """
        import asyncio

        log_always(f'FastMCPHttpClient.store_context: thread_id={thread_id}, source={source}, text_len={len(text)}')
        result = asyncio.run(self._store_context_async(thread_id, source, text, metadata))
        log_always('FastMCPHttpClient.store_context completed successfully')
        return result


def get_worktree_info(project_dir: str) -> dict[str, Any]:
    """Get worktree information for context metadata.

    Provides canonical project name from git remote URL and worktree metadata
    for proper context isolation across parallel worktree sessions.

    Fallback chain for project name:
    1. Parse repo name from git remote URL (origin -> upstream)
    2. Basename of git toplevel directory
    3. Current directory basename

    Args:
        project_dir: Project directory path.

    Returns:
        Dictionary with:
        - project: Canonical project name
        - worktree_id: Worktree directory name (None if not git repo)
        - worktree_path: Absolute path to worktree (None if not git repo)
        - is_linked_worktree: Boolean for linked worktree (None if not git repo)
    """

    def run_git(args: list[str]) -> str | None:
        """Run git command and return output, or None on failure."""
        try:
            result = subprocess.run(
                ['git', *args],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.SubprocessError, OSError):
            return None

    # Get git common dir and toplevel
    git_common_dir = run_git(['rev-parse', '--git-common-dir'])
    git_toplevel = run_git(['rev-parse', '--show-toplevel'])

    if not git_toplevel:
        # Not a git repository - use directory name as project name
        return {
            'project': Path(project_dir).name,
            'worktree_id': None,
            'worktree_path': None,
            'is_linked_worktree': None,
        }

    toplevel_path = Path(git_toplevel).resolve()

    # Detect linked worktree
    # For main worktree: git_common_dir is relative '.git' or absolute path to .git
    # For linked worktree: git_common_dir points to main repo's .git directory
    is_linked = False
    if git_common_dir:
        common_path = Path(git_common_dir)
        if common_path.is_absolute():
            expected_git_dir = toplevel_path / '.git'
            is_linked = common_path.resolve() != expected_git_dir.resolve()
        else:
            is_linked = git_common_dir not in ['.git', './.git']

    # Get canonical project name from remote URL with fallback chain
    project_name = toplevel_path.name  # Default fallback to toplevel dir name
    for remote in ['origin', 'upstream']:
        url = run_git(['remote', 'get-url', remote])
        if url:
            # Parse repo name from URL using generic pattern
            # Supports: https://github.com/user/repo.git, git@github.com:user/repo.git, etc.
            match = re.search(r'.*/([^/]+?)(?:\.git)?$', url.strip())
            if match:
                project_name = match.group(1)
                break

    return {
        'project': project_name,
        'worktree_id': toplevel_path.name,
        'worktree_path': str(toplevel_path),
        'is_linked_worktree': is_linked,
    }


def is_prebuilt_slash_command(prompt: str, config: dict[str, Any]) -> bool:
    """
    Check if a prompt is a pre-built slash command that should be skipped.

    Pre-built slash commands are built-in Claude Code commands that don't need
    context saving as they are system-level operations.

    Args:
        prompt: The user prompt to check
        config: Configuration dictionary with prebuilt_commands list

    Returns:
        True if the prompt is a pre-built slash command, False otherwise
    """
    # Get prebuilt commands from config
    prebuilt_commands = set(config.get('prebuilt_commands', DEFAULT_CONFIG['prebuilt_commands']))

    # Pattern to match slash commands at the start of a prompt
    # Matches: /command or /command with args
    # Uses \S+ to match any non-whitespace characters (handles underscores, hyphens, numbers, etc.)
    slash_command_pattern = re.compile(r'^/(\S+)(?:\s|$)')

    match = slash_command_pattern.match(prompt.strip())
    if not match:
        return False

    command_name = match.group(1).lower()
    return command_name in prebuilt_commands


def matches_skip_pattern(prompt: str, config: dict[str, Any]) -> bool:
    """
    Check if a prompt matches any skip pattern, meaning it is not the user's own words.

    UserPromptSubmit fires for more than prompts the user typed. Evaluating a hook
    of `type: prompt` triggers it with the hook's internal prompt, and Claude Code
    triggers it for turns it starts on the session's behalf: its own notifications
    (such as a finished background command or a Stop-hook wake), reports and
    messages from its subagents, messages from another Claude Code session, and
    channel-integration events. The hook input carries no field naming a turn's
    origin (anthropics/claude-code#94675 asks for one), so the only signal is the
    start of the text itself. Claude Code wraps such a turn in a tag such as
    <task-notification>, <agent-message from="...">, <cross-session-message ...>,
    or <channel source="...">, or frames a message from another session or a
    subagent with a leading sentence such as "Another Claude session sent a
    message:" or "A peer session sent a message while you were working:" ahead of
    the tagged content. A matching prompt is therefore not stored as a user
    message.

    Patterns are matched with re.match against the prompt with leading whitespace
    stripped, so a start-anchored pattern still recognizes a tag or framing
    sentence that follows a leading newline or indentation.

    Args:
        prompt: The prompt text to check
        config: Configuration dictionary with skip_patterns list

    Returns:
        True if the prompt matches any skip pattern, False otherwise
    """
    skip_patterns = config.get('skip_patterns', DEFAULT_CONFIG['skip_patterns'])

    if not skip_patterns:
        return False

    stripped_prompt = prompt.lstrip()

    for pattern in skip_patterns:
        try:
            if re.match(pattern, stripped_prompt):
                log_always(f'Prompt matches skip pattern: {pattern[:50]}...')
                return True
        except re.error as e:
            log_always(f'Invalid regex pattern "{pattern}": {e}', level='ERROR')
            continue

    return False


def skip_reason(input_data: dict[str, Any], config: dict[str, Any]) -> str | None:
    """
    Return why a hook payload is not stored, or None when its prompt is one to store.

    Only UserPromptSubmit prompts are stored, and of those neither an empty
    prompt, a pre-built slash command, nor a prompt matching a skip pattern:
    internal hook prompts and the tagged or framed turns Claude Code starts on
    the session's behalf are not the user's own words.

    Args:
        input_data: The hook payload Claude Code sent
        config: The hook configuration

    Returns:
        The log line that records why the payload is skipped, or None for a prompt to store
    """
    hook_event_name = input_data.get('hook_event_name', '')
    log_always(f'hook_event_name: {hook_event_name}')
    if hook_event_name != 'UserPromptSubmit':
        return f'Skipping: Event type is {hook_event_name}, not UserPromptSubmit'
    prompt = input_data.get('prompt', '')
    log_always(f'Prompt length: {len(prompt)} characters')
    if not prompt:
        return 'Skipping: Empty prompt'
    if is_prebuilt_slash_command(prompt, config):
        return 'Skipping: Pre-built slash command detected'
    if matches_skip_pattern(prompt, config):
        return 'Skipping: Prompt matches a skip pattern'
    return None


def resolve_thread_id(project_dir: str, config: dict[str, Any], worktree_info: dict[str, Any]) -> str:
    """
    Resolve the thread ID for context server storage.

    Discovery chain:
    1. Read .context_server/.thread_id file (with retry for Windows file-locking)
    2. Fall back to canonical project name from worktree_info

    Args:
        project_dir: The Claude project directory path
        config: Configuration dictionary with thread_id settings
        worktree_info: Pre-computed worktree information from get_worktree_info()

    Returns:
        The thread ID string (always non-empty)
    """
    thread_id_config = config.get('thread_id', DEFAULT_CONFIG['thread_id'])
    max_retries: int = thread_id_config.get('max_retries', 3)

    thread_id_file = Path(project_dir) / '.context_server' / '.thread_id'
    log_always(f'Reading thread ID from: {thread_id_file}')

    if not thread_id_file.exists():
        project_name: str = str(worktree_info['project'])
        log_always(f'Thread ID file does not exist, using project name: {project_name}')
        return project_name

    for attempt in range(max_retries):
        try:
            thread_id = thread_id_file.read_text(encoding='utf-8').strip()
            if thread_id:
                if attempt > 0:
                    log_always(f'Thread ID read succeeded on attempt {attempt + 1}/{max_retries}')
                log_always(f'Thread ID: {thread_id}')
                return thread_id
            report_error(
                'THREAD_ID_READ',
                f'Thread ID file empty (attempt {attempt + 1}/{max_retries})',
            )
        except OSError as e:
            error_msg = f'Failed to read thread ID (attempt {attempt + 1}/{max_retries}): {e}'
            if attempt < max_retries - 1:
                log_always(error_msg, level='ERROR')
                time.sleep(0.1 * (2**attempt))  # Exponential backoff: 100ms, 200ms, 400ms
            else:
                report_error('THREAD_ID_READ', error_msg)

    fallback_name: str = str(worktree_info['project'])
    log_always(f'All thread ID read attempts failed, using project name: {fallback_name}')
    return fallback_name


def create_mcp_client_with_retry(config: dict[str, Any], deadline: Deadline) -> SyncMCPClient:
    """
    Create MCP client with retry logic for uvx reliability.

    Args:
        config: Configuration dictionary with mcp_client and mcp_server settings
        deadline: The run's time budget, shared by every attempt the client makes

    Returns:
        Configured SyncMCPClient instance

    Raises:
        RuntimeError: If all retry attempts fail
    """
    mcp_config = config.get('mcp_client', DEFAULT_CONFIG['mcp_client'])
    server_config = config.get('mcp_server', DEFAULT_CONFIG['mcp_server'])

    max_retries: int = mcp_config.get('max_retries', 3)
    max_connection_retries: int = mcp_config.get('max_connection_retries', 3)
    connection_timeout: float = mcp_config.get('connection_timeout', 30.0)
    timeout_first_run: float = mcp_config.get('timeout_first_run', 120.0)
    timeout_normal: float = mcp_config.get('timeout_normal', 120.0)

    server_command: str = server_config.get('command', 'uvx')
    python_version: str = server_config.get('python_version', '3.12')
    package: str = server_config.get('package', 'mcp-context-server[embeddings-ollama]<2.0.0')
    entry_point: str = server_config.get('entry_point', 'mcp-context-server')

    log_always('Creating MCP client with retry logic')
    log_always(
        f'Config: max_retries={max_retries}, max_conn_retries={max_connection_retries}, '
        f'conn_timeout={connection_timeout}, timeout_first={timeout_first_run}, timeout_normal={timeout_normal}',
    )
    log_always(f'Server: command={server_command}, python={python_version}, package={package}, entry={entry_point}')

    last_error: Exception | None = None
    attempts = 0

    while attempts < max_retries:
        try:
            mcp_server_command = [
                server_command,
                '--python',
                python_version,
                '--with',
                package,
                entry_point,
            ]
            timeout = timeout_first_run if attempts == 0 else timeout_normal
            log_always(f'Attempt {attempts + 1}/{max_retries}: Creating MCP client (timeout={timeout}s)')
            client = SyncMCPClient(
                mcp_server_command,
                timeout=timeout,
                max_connection_retries=max_connection_retries,
                connection_timeout=connection_timeout,
                deadline=deadline,
            )

            if attempts > 0:
                log_always(f'MCP client created successfully on attempt {attempts + 1}/{max_retries}')
            else:
                log_always('MCP client created successfully')

            return client

        except Exception as e:
            last_error = e
            attempts += 1
            error_msg = f'Failed to create MCP client (attempt {attempts}/{max_retries}): {type(e).__name__}: {e}'
            log_always(error_msg, level='ERROR')

            if attempts < max_retries:
                if attempts > 1:
                    try:
                        log_always(f'{error_msg}, trying offline mode')
                        mcp_server_command = [
                            server_command,
                            '--python',
                            python_version,
                            '--with',
                            package,
                            '--offline',
                            entry_point,
                        ]
                        client = SyncMCPClient(
                            mcp_server_command,
                            timeout=timeout_normal,
                            max_connection_retries=max_connection_retries,
                            connection_timeout=connection_timeout,
                            deadline=deadline,
                        )
                        log_always('MCP client created successfully in offline mode')
                        return client
                    except Exception as offline_error:
                        log_always(
                            f'Offline mode failed: {type(offline_error).__name__}: {offline_error}',
                            level='ERROR',
                        )

                sleep_time = 0.5 * (2 ** (attempts - 1))
                log_always(f'Sleeping {sleep_time}s before retry')
                time.sleep(sleep_time)
            else:
                report_error('UVX_FAILURE', error_msg)

    if last_error:
        raise last_error
    raise RuntimeError('Failed to create MCP client after all retries')


def create_mcp_client(
    config: dict[str, Any], budget_started_at: float = _HOOK_STARTED_AT,
) -> SyncMCPClient | FastMCPHttpClient:
    """
    Create appropriate MCP client based on transport configuration.

    This factory function selects between stdio transport (a server launched through uvx)
    and HTTP transport (using FastMCP Client for remote MCP servers), and gives the
    client the run's time budget: mcp_client.deadline_seconds counted from
    budget_started_at, which caps every attempt the client makes.

    Args:
        config: Configuration dictionary with mcp_client and mcp_server settings
        budget_started_at: The time.monotonic() reading the budget counts from; the hook's own clock when omitted

    Returns:
        Either SyncMCPClient (stdio) or FastMCPHttpClient (http)

    Raises:
        ValueError: If transport type is invalid or required fields missing
    """
    server_config = config.get('mcp_server', DEFAULT_CONFIG['mcp_server'])
    mcp_config = config.get('mcp_client', DEFAULT_CONFIG['mcp_client'])
    transport = server_config.get('transport', 'stdio')

    deadline_seconds = float(mcp_config.get('deadline_seconds', DEFAULT_CONFIG['mcp_client']['deadline_seconds']))
    deadline = Deadline(deadline_seconds, started_at=budget_started_at)

    log_always(f'Creating MCP client with transport: {transport} ({deadline.describe()})')

    if transport == 'http':
        # HTTP transport using FastMCP Client
        url = server_config.get('url')
        if not url:
            raise ValueError("mcp_server.url is required when transport is 'http'")

        timeout = server_config.get('timeout', mcp_config.get('connection_timeout', 30.0))
        headers = server_config.get('headers', {})
        max_retries = mcp_config.get('max_retries', 3)

        log_always(f'HTTP transport (FastMCP): url={url}, timeout={timeout}, max_retries={max_retries}')

        return FastMCPHttpClient(
            url=url,
            timeout=timeout,
            headers=headers,
            max_retries=max_retries,
            config=config,
            deadline=deadline,
        )
    if transport == 'stdio':
        # Stdio transport via uvx subprocess. Warm the uvx cache first so the
        # initial connection attempt does not pay the package-download cost.
        _warmup_uvx_cache(server_config)
        log_always('stdio transport: using create_mcp_client_with_retry')
        return create_mcp_client_with_retry(config, deadline)
    raise ValueError(f"Invalid mcp_server.transport: {transport}. Must be 'stdio' or 'http'")


def _report_stored(result: dict[str, Any], config: dict[str, Any]) -> None:
    """
    Report a confirmed store: the context_id, or the chunk ids and any failed chunks.

    Args:
        result: The store's response, or the chunked store's summary
        config: The hook configuration
    """
    # Check for chunked storage and surface complete chunk-storage info to the
    # model via additionalContext (both success-chunk-ids AND any partial-failure
    # summary).
    if result.get('chunked', False):
        chunks_failed = result.get('chunks_failed', 0)
        total_chunks = result.get('total_chunks', 0)
        chunks_stored = result.get('chunks_stored', 0)
        failed_numbers = result.get('failed_chunk_numbers', [])

        # Derive chunk_ids by extracting context_id from each successful chunk's
        # MCP response. The store_context_chunked method returns a list of per-chunk
        # results: successful chunks have 'context_id' (per the context-server
        # store_context tool contract); failed chunks have 'error' instead.
        chunk_ids: list[int] = [
            r['context_id']
            for r in result.get('results', [])
            if 'error' not in r and r.get('context_id') is not None
        ]

        partial_failure_summary = ''
        if chunks_failed > 0:
            # Get fail_mode from config
            chunking_config = config.get('chunking', DEFAULT_CONFIG['chunking'])
            fail_mode = chunking_config.get('fail_mode', 'warn')

            partial_failure_summary = (
                f'Context storage partial failure: {chunks_failed}/{total_chunks} chunks failed '
                f'(chunks {failed_numbers}). {chunks_stored} chunks stored successfully.'
            )

            # Always log the failure (existing logging continues)
            log_always(partial_failure_summary, level='WARN')
            report_error('CHUNK_STORAGE_PARTIAL', partial_failure_summary)

            # A stderr line chosen by fail_mode. The hook exits 0, and Claude Code
            # records the stderr of a hook that exits 0 only in its debug log,
            # which exists only while its debug logging is on (for example under
            # --debug or --debug-file), so the line reaches neither the transcript
            # nor the model; the model learns of the partial failure through
            # additionalContext below while output_context_id is on.
            if fail_mode == 'warn':
                print(f'[WARN] {partial_failure_summary}', file=sys.stderr)
            elif fail_mode == 'error':
                print(f'[ERROR] {partial_failure_summary}', file=sys.stderr)
            # 'silent' mode: no stderr output, only logging

            log_always(f'User feedback mode: {fail_mode}')
        else:
            log_always(f'SUCCESS: All {total_chunks} chunks stored successfully')

        # Emit chunk-storage info to the model via additionalContext.
        # Includes chunk_ids whenever any chunks succeeded AND the partial-failure
        # summary whenever any chunks failed. Both cases coexist when partial.
        if config.get('output_context_id', True) and (chunk_ids or partial_failure_summary):
            if partial_failure_summary:
                chunk_message = (
                    f'[Hook: user message stored to context-server (chunked).'
                    f' context_ids={chunk_ids}. {partial_failure_summary}]'
                )
            else:
                chunk_message = (
                    f'[Hook: user message stored to context-server (chunked).'
                    f' context_ids={chunk_ids}]'
                )
            json_output = _load_json_output()
            json_output.emit_additional_context('UserPromptSubmit', chunk_message)
            log_always(f'Output chunk message via additionalContext: {chunk_message}')
    else:
        log_always('SUCCESS: Context stored successfully')

        # Output context_id via additionalContext for orchestrator reference
        if config.get('output_context_id', True):
            context_id = result.get('context_id')
            if context_id is not None:
                single_message = f'[Hook: user message stored to context-server. context_id={context_id}]'
                json_output = _load_json_output()
                json_output.emit_additional_context('UserPromptSubmit', single_message)
                log_always(f'Output context_id={context_id} via additionalContext')


def _unconfirmed_store_notice(error: Exception, error_msg: str, thread_id: str, metadata: dict[str, Any]) -> str:
    """
    Compose what the model is told when the message was not confirmed stored.

    The model has the context-server tools, so it can store the prompt itself.
    Whether it should do so now depends on the stage the run reached. Every
    failure that follows a sent store request arrives as an UnconfirmedStoreError
    with request_sent set, so any other error means nothing was sent: the thread
    holds no copy, and the model stores the prompt right away. After a request
    was sent the server may still complete it, and a copy stored now would
    duplicate it, so the model searches the thread at the end of the turn, when
    the server has had the turn to finish, and stores the prompt only if no
    entry holds it; a chunked store names the chunks already stored.

    Args:
        error: The error that ended the store
        error_msg: The error's type and message, as the notice shows them
        thread_id: The thread the message belongs in
        metadata: The metadata the hook would have stored the message with

    Returns:
        The notice text for additionalContext
    """
    store_call = f'the context-server store_context tool (thread_id={thread_id}, source=user, metadata={json.dumps(metadata)})'
    if not (isinstance(error, UnconfirmedStoreError) and error.request_sent):
        return (
            f'[Hook: user message not stored to context-server (thread_id={thread_id}; {error_msg}).'
            f' No store request was sent, so the thread holds no copy: store the prompt yourself now with {store_call}.]'
        )
    if error.total_chunks is not None:
        stage = (
            f' The chunked store was cut short: the server confirmed {len(error.stored_context_ids)} of'
            f' {error.total_chunks} chunks (context_ids={error.stored_context_ids}) and may still complete the chunk'
            f' in flight, so do not store a copy now.'
            f' At the end of this turn, search thread {thread_id} for the text of this prompt,'
            f' and store only what no entry holds yourself with {store_call}.]'
        )
    else:
        stage = (
            ' The store request reached the server, which may still complete it, so do not store a copy now.'
            f' At the end of this turn, search thread {thread_id} for the text of this prompt, and only if no entry holds it,'
            f' store the prompt yourself with {store_call}.]'
        )
    return f'[Hook: user message not confirmed stored to context-server (thread_id={thread_id}; {error_msg}).{stage}'


def hook_clock() -> float:
    """Return the hook's own clock: the time.monotonic() reading taken when its module code started."""
    return _HOOK_STARTED_AT


def main(
    launch_clock: Callable[[], float] = hook_clock,
    trace: StartupTrace = DISABLED_TRACE,
    read_start_times: Callable[[], StartTimes] = own_start_times,
) -> None:
    """
    Main hook execution function.

    The payload is checked against the filters before any of the store's work,
    because a turn that is not stored needs only its payload, the configuration,
    and the skip patterns: no process is read before that decision, and the
    launch is read and fastmcp imported only for a prompt to store. Right after
    the decision, on both paths, a run whose trace records reads its start
    times and records them as the start_times stage; a creation time does not
    change, so reading it late loses nothing, and a run killed in that read
    leaves a trace ending at skip_decided.

    Args:
        launch_clock: Returns the time.monotonic() reading the store's budget counts from; called only for a
            prompt to store, and the hook's own clock when omitted
        trace: The run's startup trace, which records each stage main() reaches
        read_start_times: Returns when the hook's interpreter, and the venv redirector that started it, were
            created; called only while the trace records, and read from the platform's process table when omitted
    """
    log_always('Entering main() function')
    start_time = datetime.now(tz=UTC)

    # Load configuration (defaults merged with config file if provided)
    try:
        config_loader = _load_config_loader()
        config: dict[str, Any] = config_loader.get_config_from_argv(DEFAULT_CONFIG)
    except Exception:
        # If config loading fails, use defaults
        config = DEFAULT_CONFIG.copy()
    trace.record('config_loaded')

    # Check if hook is enabled
    if not config.get('enabled', True):
        log_always('Hook disabled via config, exiting')
        sys.exit(0)

    # Claude Code writes the payload as UTF-8. The stdin text layer decodes with the locale's
    # encoding unless Python runs in UTF-8 mode, so the payload is parsed from the raw bytes.
    log_always('Reading stdin data')
    try:
        input_data = json.load(sys.stdin.buffer)
        log_always(f'stdin data keys: {list(input_data.keys())}')
    except json.JSONDecodeError as e:
        log_always(f'JSON decode error: {e}', level='ERROR')
        log_always('Exiting: Invalid JSON from stdin')
        sys.exit(0)
    except Exception as e:
        log_always(f'Error reading stdin: {type(e).__name__}: {e}', level='ERROR')
        log_always('Exiting: Failed to read stdin')
        sys.exit(0)
    trace.record('stdin_read')

    reason = skip_reason(input_data, config)
    trace.record('skip_decided', store=reason is None)
    if trace.recording:
        times = read_start_times()
        trace.record(
            'start_times',
            interpreter_created=None if times.interpreter is None else _utc_text(times.interpreter),
            redirector_created=None if times.redirector is None else _utc_text(times.redirector),
        )
    if reason is not None:
        log_always(reason)
        sys.exit(0)
    prompt: str = input_data['prompt']

    # CRITICAL: Configure UTF-8 for Windows BEFORE any subprocess operations
    # This prevents non-ASCII text corruption (mojibake) in MCP communication
    setup_windows_utf8()

    # Get Claude project directory
    claude_project_dir = os.environ.get('CLAUDE_PROJECT_DIR')
    log_always(f'CLAUDE_PROJECT_DIR for context save: {claude_project_dir}')
    if not claude_project_dir:
        # No project directory, can't proceed
        log_always('Exiting: CLAUDE_PROJECT_DIR not set', level='ERROR')
        sys.exit(0)

    # Get worktree information for context metadata and thread ID resolution
    # Provides canonical project name from git remote URL for cross-worktree consistency
    worktree_info = get_worktree_info(claude_project_dir)
    log_always(f'Worktree info: project={worktree_info["project"]}')

    # Resolve thread ID (reads .thread_id file, falls back to project name)
    thread_id = resolve_thread_id(claude_project_dir, config, worktree_info)

    # Build metadata with worktree fields for context isolation.
    # kind and schema_version follow the universal context-server metadata
    # schema v1: every entry carries a record-kind discriminator so filters
    # and future server-side schema enforcement can rely on it.
    metadata: dict[str, Any] = {
        'schema_version': 1,
        'kind': 'user_message',
        'project': worktree_info['project'],
    }
    # Add optional worktree fields if available (git repository detected)
    if worktree_info.get('worktree_id') is not None:
        metadata['worktree_id'] = worktree_info['worktree_id']
    if worktree_info.get('worktree_path') is not None:
        metadata['worktree_path'] = worktree_info['worktree_path']
    if worktree_info.get('is_linked_worktree') is not None:
        metadata['is_linked_worktree'] = worktree_info['is_linked_worktree']
    trace.record('project_resolved')

    budget_started_at = launch_clock()
    log_always(f'Run budget counts from {_HOOK_STARTED_AT - budget_started_at:.2f}s before the hook clock started')
    launcher_created = trace.wall_text(budget_started_at) if budget_started_at < _HOOK_STARTED_AT else None
    trace.record('launcher_read', launcher_created=launcher_created)

    # Loaded once the prompt is known to be stored, so a skipped turn never pays for it, and outside
    # the store's error handling, so a broken installation fails the run instead of reading as an
    # unconfirmed store; the clients take Client and their transports from the loaded package
    trace.record('fastmcp_import_started')
    importlib.import_module('fastmcp')
    log_always('FastMCP imported')
    trace.record('fastmcp_imported')

    # Create MCP client and store context
    trace.record('store_started')
    try:
        # A failure in MCP communication never blocks the prompt: the hook exits 0
        # and tells the model below that the message was not confirmed stored
        #
        # CRITICAL UTF-8 REQUIREMENT:
        # The setup_windows_utf8() function MUST be called before this point
        # to ensure subprocess stdin/stdout use UTF-8 encoding.
        # Without this, non-ASCII text (Cyrillic, Chinese, Arabic, etc.) gets
        # corrupted on Windows due to codepage defaults (CP1252/Windows-1251).
        #
        # FastMCP's StdioTransport does not explicitly set encoding='utf-8'
        # on subprocess pipes, so we configure it via environment variable
        # (PYTHONUTF8=1) and console codepage (65001) instead.

        # Create client based on transport configuration (stdio or http)
        client = create_mcp_client(config, budget_started_at)

        log_always('Storing context in MCP server')
        result = client.store_context(
            thread_id=thread_id,
            source='user',
            text=prompt,
            metadata=metadata,
        )
    except Exception as e:
        trace.record('store_ended', confirmed=False)
        # Log the error with its traceback, then tell the model instead of blocking the prompt
        cause = e.cause if isinstance(e, UnconfirmedStoreError) else e
        error_msg = f'{type(cause).__name__}: {cause}'
        full_traceback = traceback.format_exc()

        # Check for specific error patterns related to pipe buffer issues
        error_str = str(e).lower()
        error_context = ''

        if 'broken pipe' in error_str:
            error_context = ' (Message likely too large for subprocess pipe buffer)'
        elif 'timed out' in error_str or 'deadline' in error_str:
            error_context = ' (The store did not finish within the run budget; see mcp_client.deadline_seconds)'
        elif '[errno 32]' in error_str or 'epipe' in error_str:
            error_context = ' (Subprocess pipe broken - message size may exceed buffer capacity)'
        elif 'buffer' in error_str:
            error_context = ' (Buffer-related error - message may be too large)'

        log_always(f'MCP store failure: {error_msg}{error_context}', level='ERROR')
        log_always(f'Traceback:\n{full_traceback}', level='ERROR')
        report_error('MCP_STORE_FAILURE', f'{error_msg}{error_context}\n{full_traceback}')

        failure_message = _unconfirmed_store_notice(e, error_msg, thread_id, metadata)
        json_output = _load_json_output()
        json_output.emit_additional_context('UserPromptSubmit', failure_message)
        log_always(f'Output store failure notice via additionalContext: {failure_message}')
    else:
        trace.record('store_ended', confirmed=True)
        _report_stored(result, config)

    # Always exit successfully
    end_time = datetime.now(tz=UTC)
    duration = (end_time - start_time).total_seconds()
    log_always(f'Execution completed in {duration:.3f} seconds')
    sys.exit(0)


def entry() -> None:
    """
    Run the hook as Claude Code starts it.

    The run's startup trace begins first, naming the process and the
    configuration file, and is removed when main() ends the run, which it does
    through sys.exit; a run that is killed, or that main() leaves with an
    unexpected error, keeps it. The process table is read only once main() has
    decided the skip: for the start times while the trace records, and for the
    launch, which sets the store's budget, only if main() reaches the store.

    Raises:
        SystemExit: The exit main() ends the run with
    """
    pid = os.getpid()
    trace = StartupTrace.begin(
        startup_trace_directory(),
        _HOOK_STARTED_WALL,
        _HOOK_STARTED_AT,
        pid=pid,
        config=sys.argv[1] if len(sys.argv) > 1 else None,
    )
    try:
        main(lambda: launcher_clock_start(_HOOK_STARTED_AT, process_table(), sys.executable, pid), trace)
    except SystemExit:
        trace.finish()
        raise
    trace.finish()


if __name__ == '__main__':
    entry()

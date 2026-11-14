"""Database URL helpers shared by the ``mcp-context-server-migrate`` modes.

Classifies a ``--source-url``/``--target-url`` value as SQLite or PostgreSQL and
masks the password of a PostgreSQL URL before it is printed or logged.
"""

import re
from urllib.parse import urlparse


def parse_backend_url(url: str) -> tuple[str, str]:
    """Classify a database URL and return ``(backend_type, address)``.

    Backend type is one of ``"sqlite"`` or ``"postgresql"``. The address
    form depends on the backend:

    - ``sqlite``: filesystem path (absolute or relative).
    - ``postgresql``: the original URL, suitable for ``asyncpg.connect``.

    Recognition rules:

    - URL starting with ``sqlite://`` or ``sqlite:`` is SQLite.
    - URL starting with ``postgresql://`` or ``postgres://`` is
      PostgreSQL.
    - URL with no scheme and a path-like value is treated as SQLite.

    Args:
        url: The database URL or filesystem path.

    Returns:
        Tuple of ``(backend_type, address)``.

    Raises:
        ValueError: If ``url`` cannot be classified.
    """
    lowered = url.lower().strip()
    if not lowered:
        raise ValueError('database URL must not be empty')
    if lowered.startswith('sqlite://'):
        path = url[len('sqlite://') :]
        if path.startswith('//'):
            # SQLAlchemy POSIX absolute form: sqlite:////abs/path keeps an
            # extra leading slash after the scheme strip. Collapse the run to
            # a single slash.
            path = '/' + path.lstrip('/')
        if path.startswith('/') and len(path) >= 3 and path[2] == ':':
            # Windows drive form: sqlite:///C:/foo -> C:/foo
            path = path.lstrip('/')
        return ('sqlite', path)
    if lowered.startswith('sqlite:'):
        return ('sqlite', url[len('sqlite:') :])
    if lowered.startswith(('postgresql://', 'postgres://')):
        return ('postgresql', url)
    # Bare Windows absolute path (e.g. ``C:\path\db`` or ``C:/path/db``).
    # ``urlparse`` would misread the single-letter drive as a URL scheme and
    # reject it, so detect it explicitly and treat it as a SQLite filesystem
    # path -- the CLI accepts plain paths without a scheme on every platform.
    if re.match(r'^[A-Za-z]:[\\/]', url):
        return ('sqlite', url)
    parsed = urlparse(url)
    if parsed.scheme in ('', 'file'):
        if parsed.scheme == 'file':
            return ('sqlite', parsed.path)
        return ('sqlite', url)
    raise ValueError(f'Unrecognized database URL scheme: {url!r}')


_POSTGRESQL_CREDENTIAL_RE = re.compile(r'(postgres(?:ql)?://[^:@/]*):[^@/]*@', re.IGNORECASE)


def mask_credentials(url: str) -> str:
    """Mask the password portion of a PostgreSQL URL.

    SQLite paths are returned unchanged. PostgreSQL URLs of the form
    ``postgresql://user:password@host/db`` have ``password`` replaced by
    ``***``.

    Args:
        url: The original URL.

    Returns:
        Same URL with the password segment redacted.
    """
    return _POSTGRESQL_CREDENTIAL_RE.sub(r'\1:***@', url)

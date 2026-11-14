"""Tests for the database URL helpers shared by the migration CLI modes."""

import pytest

from app.cli._database_url import mask_credentials
from app.cli._database_url import parse_backend_url


class TestUrlHelpers:
    """URL parsing and credential masking helpers."""

    def test_parse_backend_url_sqlite_form(self) -> None:
        """``sqlite://`` URLs resolve to a SQLite address."""
        kind, addr = parse_backend_url('sqlite:///tmp/file.db')
        assert kind == 'sqlite'
        assert addr.endswith('tmp/file.db')

    def test_parse_backend_url_sqlalchemy_posix_absolute_form(self) -> None:
        """``sqlite:////abs/path`` collapses to a single-slash absolute path.

        The SQLAlchemy absolute form on POSIX uses four slashes; keeping the
        double-slash prefix would later be parsed as a file-URI authority by
        the SQLite backend and rejected.
        """
        kind, addr = parse_backend_url('sqlite:////tmp/file.db')
        assert kind == 'sqlite'
        assert addr == '/tmp/file.db'

    def test_parse_backend_url_windows_drive_scheme_form(self) -> None:
        """``sqlite:///C:/foo`` strips the leading slash before the drive letter."""
        kind, addr = parse_backend_url('sqlite:///C:/data/file.db')
        assert kind == 'sqlite'
        assert addr == 'C:/data/file.db'

    def test_parse_backend_url_postgresql_form(self) -> None:
        """``postgresql://`` URLs resolve to a PostgreSQL address."""
        kind, addr = parse_backend_url('postgresql://u:p@h/db')
        assert kind == 'postgresql'
        assert addr == 'postgresql://u:p@h/db'

    def test_parse_backend_url_windows_backslash_path(self) -> None:
        """A bare Windows absolute path with backslashes is treated as SQLite.

        ``urlparse`` would misread the ``C:`` drive letter as a URL scheme; the
        parser must recognize the drive-letter form and return it verbatim.
        """
        win_path = 'C:\\Users\\me\\AppData\\Local\\Temp\\v2_source.db'
        kind, addr = parse_backend_url(win_path)
        assert kind == 'sqlite'
        assert addr == win_path

    def test_parse_backend_url_windows_forwardslash_path(self) -> None:
        """A bare Windows absolute path with forward slashes is treated as SQLite."""
        kind, addr = parse_backend_url('D:/data/v3_target.db')
        assert kind == 'sqlite'
        assert addr == 'D:/data/v3_target.db'

    def test_parse_backend_url_posix_path(self) -> None:
        """A bare POSIX absolute path is treated as SQLite."""
        kind, addr = parse_backend_url('/home/me/db.sqlite')
        assert kind == 'sqlite'
        assert addr == '/home/me/db.sqlite'

    @pytest.mark.parametrize(
        ('url', 'expected'),
        [
            ('postgres://u:p@h/db', ('postgresql', 'postgres://u:p@h/db')),
            ('sqlite:data/file.db', ('sqlite', 'data/file.db')),
            ('data/file.db', ('sqlite', 'data/file.db')),
            ('file:///tmp/file.db', ('sqlite', '/tmp/file.db')),
        ],
    )
    def test_parse_backend_url_other_accepted_forms(self, url: str, expected: tuple[str, str]) -> None:
        """The short PostgreSQL scheme, a scheme without slashes, a relative path and a file URI are accepted."""
        assert parse_backend_url(url) == expected

    def test_mask_credentials_redacts_password(self) -> None:
        """The password segment of a PostgreSQL URL is masked."""
        masked = mask_credentials('postgresql://user:secret@host/db')
        assert 'secret' not in masked
        assert 'user' in masked
        assert '***' in masked

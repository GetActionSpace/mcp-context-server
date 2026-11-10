"""The connection strings of one row-level security test database, one per role (:class:`RlsUrls`).

The ``pg_rls_db`` fixture of the package's ``conftest.py`` provisions the database and hands
out an :class:`RlsUrls`; the modules that type a parameter with it import it from here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RlsUrls:
    """Connection strings of one row-level security test database, one per role.

    Attributes:
        admin: ``mcp_test``, the bootstrap superuser: database administration and the
            oracle channel that reads every row whatever the policies say.
        maintenance: ``mcp_owner``, the non-superuser login role that owns the database
            and therefore every table the maintenance backend creates in it.
        app: ``mcp_app``, the request login role, an inheriting member of the policy role
            ``mcp_context_request``.
        outsider: ``mcp_outsider``, a login role that is a member of no role.
    """

    admin: str
    maintenance: str
    app: str
    outsider: str

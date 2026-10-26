"""The two-principal case registry: the cases of every seam group in one tuple.

Each seam group keeps its cases in its own ``_access_scope_cases_<group>.py`` module, built
on the shared scaffold of :mod:`tests.repositories._access_scope_cases`. This module only
collects them, so the scaffold never imports a group and no import cycle forms. Both entry
points parametrize over :data:`CASES`; the PostgreSQL entry point also runs
:data:`RLS_CASES` under row-level security and :data:`READ_DECIDED_CASES` with the database
as the only filter.
"""

from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases_aggregates import AGGREGATE_CASES
from tests.repositories._access_scope_cases_dedup import DEDUP_CASES
from tests.repositories._access_scope_cases_deletes import DELETE_CASES
from tests.repositories._access_scope_cases_generation_jobs import GENERATION_JOB_READ_CASES
from tests.repositories._access_scope_cases_generation_jobs import GENERATION_JOB_WRITE_CASES
from tests.repositories._access_scope_cases_generation_jobs import RLS_GENERATION_JOB_CASES
from tests.repositories._access_scope_cases_grants import GRANT_CASES
from tests.repositories._access_scope_cases_ingest import INGEST_CASES
from tests.repositories._access_scope_cases_ingest import INGEST_OWNER_CASES
from tests.repositories._access_scope_cases_ranked import RANKED_CASES
from tests.repositories._access_scope_cases_reads import READ_CASES
from tests.repositories._access_scope_cases_search import SEARCH_CASES
from tests.repositories._access_scope_cases_updates import RLS_UPDATE_CASES
from tests.repositories._access_scope_cases_updates import UPDATE_CASES

# The registered cases, each proving one seam for every scope it names.
CASES: tuple[AccessCase, ...] = (
    *DEDUP_CASES, *READ_CASES, *SEARCH_CASES, *UPDATE_CASES, *DELETE_CASES, *RANKED_CASES, *AGGREGATE_CASES,
    *INGEST_CASES, *INGEST_OWNER_CASES, *GRANT_CASES, *GENERATION_JOB_READ_CASES, *GENERATION_JOB_WRITE_CASES,
)

# The cases that only a database enforcing row-level security runs, beside every case of CASES.
RLS_CASES: tuple[AccessCase, ...] = (*RLS_UPDATE_CASES, *RLS_GENERATION_JOB_CASES)

# The cases of the read, search, ranked, aggregate, ingest-range and generation job read groups
# whose observable the rows each scope may read decide alone, so the database's answer for the
# scope's identity must reproduce it without the application predicate. The owned ingest listing
# is decided by ownership, which the database's read view does not decide, so its cases stay out.
READ_DECIDED_CASES: tuple[AccessCase, ...] = tuple(
    case
    for case in (*READ_CASES, *SEARCH_CASES, *RANKED_CASES, *AGGREGATE_CASES, *INGEST_CASES, *GENERATION_JOB_READ_CASES)
    if not case.reports_write_access
)

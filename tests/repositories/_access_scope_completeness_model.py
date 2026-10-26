"""The classification model of the access-scope completeness tests, read from the source of ``app/``.

The unit of classification is the method: a function defined directly under a class or a
module of ``app/repositories``, with every closure nested inside it counted as part of it.
:data:`REGISTRY` assigns every repository method that runs SQL against ``context_entries`` or
one of its child tables a :class:`Seam`, whether the statement names the table itself or reads
it through the read seam ``_readable``, and the helpers below parse ``app/`` once and answer
the structural questions the completeness tests ask of it, among them where each child reader,
child writer and worker method is referenced (:func:`qualified_references`,
:func:`worker_site_violations`).
"""

import ast
import functools
import importlib
import inspect
import re
import typing
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'app'

CONTEXT_TABLE_SQL = re.compile(
    r'\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+(context_entries|context_entries_fts|context_entries_fts_docsize'
    r'|context_entry_grants|tags|image_attachments|context_index_nodes|embedding_chunks|embedding_metadata'
    r'|vec_context_embeddings|vec_context_embeddings_compressed|generation_jobs)\b',
    re.IGNORECASE,
)

# The repository method a plain read names each routable relation through; the backend
# renders the relation, so the table appears in the call rather than in the statement text.
READ_SEAM = '_readable'


class Seam(StrEnum):
    """How a method or a ranked-search closure is guarded."""

    SCOPED = 'scoped'  # the read predicate sits in the statement
    WRITE_AUTHORIZED = 'write_authorized'  # the write or owner predicate sits in the statement
    # an owner predicate keyed on a caller principal that also stamps or matches it: the dedup fragments
    # and the grant statements
    OWNER_KEYED = 'owner_keyed'
    OWNER_STAMPED = 'owner_stamped'  # an insert whose owner_id is bound from the scope; no predicate
    CHILD_BY_TRUSTED_ID = 'child_by_trusted_id'  # id-keyed child read; its call sites are pinned
    CHILD_WRITER = 'child_writer'  # child write behind the write gate or the owned dedup; sites pinned
    # a job-queue or generated-output statement that runs under the identity its caller passes, with no
    # predicate: outputs are generated for whichever entry holds a job, and enqueuing the job was authorized;
    # only the generation worker, the CLI and other worker methods reference it
    WORKER = 'worker'
    SYSTEM = 'system'  # schema maintenance or a deployment-level figure
    COMPUTE = 'compute'  # a ranked-search closure that runs no SQL


# The classes whose methods render the access predicate.
PREDICATE_SEAMS = frozenset({Seam.SCOPED, Seam.WRITE_AUTHORIZED, Seam.OWNER_KEYED})
GUARDED = PREDICATE_SEAMS | {Seam.OWNER_STAMPED}
CHILD = frozenset({Seam.CHILD_BY_TRUSTED_ID, Seam.CHILD_WRITER})

# The statement builders an owner-stamped method may insert through instead of holding the INSERT itself.
OWNER_STAMPED_BUILDERS = frozenset({'_insert_entry_sql'})

# The top-level directories of app/ whose code may call a worker method: the generation worker
# and the command-line tools, which drain the job queue outside any request.
WORKER_CALLER_HOMES = frozenset({'generation_jobs', 'cli'})


@dataclass(frozen=True)
class Entry:
    """The classification of one repository method.

    Attributes:
        seam: How the method is guarded.
        cases: The two-principal case ids that prove a guarded method on both backends.
        closures: For the two ranked searches, the class of every closure defined directly
            inside the method.
        forwards_to: For a dispatcher, the executors it hands its scope to.
        spies: For a dispatcher, the tests (``path::name``) that spy on every branch.
        sites: For a child reader or writer, every function of ``app/`` that references it,
            as ``<path under app>:<qualified function name>``, once per reference.
        receiver: Counts only references through this attribute or name, for a method whose
            name other objects share (``Path.exists``, a circuit breaker's ``record_failure``).
    """

    seam: Seam
    cases: tuple[str, ...] = ()
    closures: Mapping[str, Seam] = field(default_factory=dict[str, Seam])
    forwards_to: tuple[str, ...] = ()
    spies: tuple[str, ...] = ()
    sites: tuple[str, ...] = ()
    receiver: str | None = None


_REPO = 'app.repositories.'
_READS = _REPO + 'context_repository.reads:ContextReadMixin.'
_SEARCH = _REPO + 'context_repository.search:ContextSearchMixin.'
_UPDATES = _REPO + 'context_repository.updates:ContextUpdateMixin.'
_DELETES = _REPO + 'context_repository.deletes:ContextDeleteMixin.'
_DEDUP = _REPO + 'context_repository.dedup:'
_INGEST = _REPO + 'context_repository.ingest:ContextIngestMixin.'
_EMBEDDINGS = _REPO + 'embedding_repository:EmbeddingRepository.'
_FP32 = _REPO + 'embedding_repository.fp32_search:Fp32SearchMixin.'
_COMPRESSED = _REPO + 'embedding_repository.compressed_search:CompressedSearchMixin.'
_INVENTORY = _REPO + 'embedding_repository.inventory:EmbeddingInventoryMixin.'
_CHUNKS = _REPO + 'embedding_repository.chunk_writes:ChunkWriteMixin.'
_FTS_SEARCH = _REPO + 'fts_repository.search:FtsSearchMixin.'
_FTS_MAINTENANCE = _REPO + 'fts_repository.maintenance:FtsMaintenanceMixin.'
_STATISTICS = _REPO + 'statistics_repository:StatisticsRepository.'
_NODES = _REPO + 'index_node_repository:IndexNodeRepository.'
_TAGS = _REPO + 'tag_repository:TagRepository.'
_IMAGES = _REPO + 'image_repository:ImageRepository.'
_GRANTS = _REPO + 'grant_repository:GrantRepository.'
_JOBS = _REPO + 'generation_job_repository:GenerationJobRepository.'
_GENERATION = _REPO + 'context_repository.generation:ContextGenerationMixin.'

_STORE_TX = 'tools/_transactions.py:execute_store_in_transaction'
_UPDATE_TX = 'tools/_transactions.py:execute_update_in_transaction'
_CHUNK_WRITES = 'repositories/embedding_repository/chunk_writes.py:ChunkWriteMixin.'
_REEMBED = 'cli/migrate_reembed.py:_reembed_async'
_RESULT_HYDRATORS = (
    'tools/context/retrieve.py:get_context_by_ids',
    'tools/search/browse.py:search_context',
    'tools/search/fts.py:fts_search_context',
    'tools/search/semantic.py:semantic_search_context',
    'tools/search/hybrid.py:hybrid_search_context',
)
_EMBEDDING_DISPATCH_SPIES = 'tests/repositories/embedding_repository/test_search_dispatch.py::'
_FTS_DISPATCH_SPIES = 'tests/repositories/fts_repository/test_search_dispatch.py::'


def _guarded(
    seam: Seam,
    *cases: str,
    closures: Mapping[str, Seam] | None = None,
    forwards_to: tuple[str, ...] = (),
    spies: tuple[str, ...] = (),
) -> Entry:
    """Build the entry of a guarded method proven by ``cases``."""
    return Entry(seam, cases=cases, closures=closures or {}, forwards_to=forwards_to, spies=spies)


def _child(seam: Seam, *sites: str, receiver: str | None = None, cases: tuple[str, ...] = ()) -> Entry:
    """Build the entry of a child reader or writer referenced from ``sites`` and proven by ``cases``."""
    return Entry(seam, cases=cases, sites=sites, receiver=receiver)


def _worker(*, receiver: str | None = None) -> Entry:
    """Build the entry of a worker method, whose references :func:`worker_site_violations` checks."""
    return Entry(Seam.WORKER, receiver=receiver)


REGISTRY: dict[str, Entry] = {
    # Scoped: the read predicate in the statement.
    _READS + 'get_by_ids': _guarded(Seam.SCOPED, 'S3'),
    _READS + 'find_ids_by_prefix': _guarded(Seam.SCOPED, 'S4'),
    _READS + 'check_entry_exists': _guarded(Seam.SCOPED, 'S5'),
    _READS + 'probe_ids': _guarded(Seam.SCOPED, 'S7'),
    _SEARCH + 'search_contexts': _guarded(Seam.SCOPED, 'S1'),
    _SEARCH + 'grep_scan_text_contents': _guarded(Seam.SCOPED, 'S2'),
    _SEARCH + '_build_context_filter_clause': _guarded(Seam.SCOPED, 'S1', 'S2'),
    _EMBEDDINGS + 'search': _guarded(
        Seam.SCOPED, 'R1', 'R2', forwards_to=('search_fp32', 'search_compressed'),
        spies=(
            _EMBEDDING_DISPATCH_SPIES + 'test_compressed_branch_receives_the_scope',
            _EMBEDDING_DISPATCH_SPIES + 'test_fp32_branch_receives_the_scope',
        ),
    ),
    _FP32 + 'search_fp32': _guarded(
        Seam.SCOPED, 'R1', closures={'_search_sqlite': Seam.SCOPED, '_search_postgresql': Seam.SCOPED},
    ),
    _COMPRESSED + 'search_compressed': _guarded(
        Seam.SCOPED, 'R2', 'R3', closures={
            '_candidates_sqlite': Seam.SCOPED,
            '_candidates_pg': Seam.SCOPED,
            '_hydrate_sqlite': Seam.SCOPED,
            '_hydrate_pg': Seam.SCOPED,
            '_read_compressed_sqlite': Seam.CHILD_BY_TRUSTED_ID,
            '_read_compressed_pg': Seam.CHILD_BY_TRUSTED_ID,
            '_decode_and_concat': Seam.COMPUTE,
            '_rank_from_gemm': Seam.COMPUTE,
        },
    ),
    _FTS_SEARCH + 'search': _guarded(
        Seam.SCOPED, 'R4', forwards_to=('_search_sqlite', '_search_postgresql'),
        spies=(_FTS_DISPATCH_SPIES + 'test_executor_receives_the_scope',),
    ),
    _FTS_SEARCH + '_search_sqlite': _guarded(Seam.SCOPED, 'R4'),
    _FTS_SEARCH + '_search_postgresql': _guarded(Seam.SCOPED, 'R4'),
    _FTS_MAINTENANCE + 'get_statistics': _guarded(Seam.SCOPED, 'A5'),
    _INVENTORY + 'get_statistics': _guarded(Seam.SCOPED, 'A4'),
    _STATISTICS + 'get_thread_list': _guarded(Seam.SCOPED, 'A1'),
    _STATISTICS + 'get_database_statistics': _guarded(Seam.SCOPED, 'A2', 'J1'),
    _STATISTICS + 'get_summary_statistics': _guarded(Seam.SCOPED, 'A3'),
    _NODES + 'count_all_nodes': _guarded(Seam.SCOPED, 'A6'),
    _INGEST + 'ingest_rows_in_range': _guarded(Seam.SCOPED, 'S8'),
    # Write-authorized: the write or owner predicate in the statement.
    _READS + 'entry_exists': _guarded(Seam.WRITE_AUTHORIZED, 'S6'),
    _READS + 'get_content_type': _guarded(Seam.WRITE_AUTHORIZED, 'W5'),
    _UPDATES + 'update_context_entry': _guarded(Seam.WRITE_AUTHORIZED, 'W1', 'W2'),
    _UPDATES + 'patch_metadata': _guarded(Seam.WRITE_AUTHORIZED, 'W3'),
    _UPDATES + 'touch_updated_at': _guarded(Seam.WRITE_AUTHORIZED, 'W4'),
    _UPDATES + 'update_content_type': _guarded(Seam.WRITE_AUTHORIZED, 'W4'),
    _DELETES + 'delete_by_ids': _guarded(Seam.WRITE_AUTHORIZED, 'X1'),
    _DELETES + 'get_ids_matching_batch_criteria': _guarded(Seam.WRITE_AUTHORIZED, 'X2'),
    _INGEST + 'owned_ingest_page': _guarded(Seam.WRITE_AUTHORIZED, 'X3'),
    # Owner-keyed: the dedup candidate, the read-scoped interleave check and the owned update.
    _DEDUP + 'ContextDedupMixin.store_with_deduplication': _guarded(Seam.OWNER_KEYED, 'K1', 'K2', 'K4'),
    _DEDUP + 'ContextDedupMixin.check_latest_is_duplicate': _guarded(Seam.OWNER_KEYED, 'K3'),
    _DEDUP + '_candidate_sql': _guarded(Seam.OWNER_KEYED, 'K1', 'K3', 'K4'),
    _DEDUP + '_interleave_sql': _guarded(Seam.OWNER_KEYED, 'K2', 'K3'),
    _DEDUP + '_dedup_update_sql': _guarded(Seam.OWNER_KEYED, 'K1'),
    # Owner-keyed: the grant statements, which act only on an entry the caller owns; an insert records the
    # caller as grantor.
    _GRANTS + 'store_group_read_grants': _guarded(Seam.OWNER_KEYED, 'G1'),
    _GRANTS + '_grant_insert_sql': _guarded(Seam.OWNER_KEYED, 'G1', 'G2'),
    _GRANTS + 'store_grant': _guarded(Seam.OWNER_KEYED, 'G3'),
    _GRANTS + 'delete_grant': _guarded(Seam.OWNER_KEYED, 'G4'),
    _GRANTS + 'list_grants': _guarded(Seam.OWNER_KEYED, 'G5'),
    # Owner-stamped: a fresh insert owned by the scope's principal.
    _DEDUP + '_insert_entry_sql': _guarded(Seam.OWNER_STAMPED, 'K1', 'K5'),
    _DEDUP + 'ContextDedupMixin.insert_entry': _guarded(Seam.OWNER_STAMPED, 'K5'),
    # Child readers keyed by ids a scoped read produced in the same request.
    _TAGS + 'get_tags_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, *_RESULT_HYDRATORS),
    _IMAGES + 'get_images_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, *_RESULT_HYDRATORS),
    _IMAGES + 'count_images_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, _UPDATE_TX),
    _NODES + 'get_nodes_for_context': _child(Seam.CHILD_BY_TRUSTED_ID, 'tools/navigation/navigate.py:navigate_context'),
    _JOBS + 'states_for_ids': _child(Seam.CHILD_BY_TRUSTED_ID, cases=('J2',)),
    _JOBS + 'state_for': _child(Seam.CHILD_BY_TRUSTED_ID),
    _JOBS + '_states_sql': _child(
        Seam.CHILD_BY_TRUSTED_ID,
        'repositories/generation_job_repository.py:GenerationJobRepository.state_for',
        'repositories/generation_job_repository.py:GenerationJobRepository.states_for_ids._read_sqlite',
        'repositories/generation_job_repository.py:GenerationJobRepository.states_for_ids._read_postgresql',
    ),
    _INVENTORY + 'exists': _child(
        Seam.CHILD_BY_TRUSTED_ID,
        _STORE_TX, 'tools/batch/store.py:store_context_batch', 'tools/context/store.py:store_context',
        receiver='embeddings',
    ),
    # Child writers behind the write gate or the owned dedup.
    _TAGS + 'store_tags': _child(Seam.CHILD_WRITER, _STORE_TX),
    _TAGS + 'replace_tags_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX),
    _TAGS + '_insert_tag_sql': _child(
        Seam.CHILD_WRITER,
        'repositories/tag_repository.py:TagRepository.store_tags._store_tags_sqlite',
        'repositories/tag_repository.py:TagRepository.store_tags._store_tags_postgresql',
        'repositories/tag_repository.py:TagRepository.replace_tags_for_context._replace_tags_sqlite',
        'repositories/tag_repository.py:TagRepository.replace_tags_for_context._replace_tags_postgresql',
    ),
    _IMAGES + 'store_images': _child(Seam.CHILD_WRITER, _STORE_TX),
    _IMAGES + 'replace_images_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX, _UPDATE_TX),
    _NODES + 'replace_nodes_for_context': _child(Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX),
    _JOBS + 'enqueue': _child(Seam.CHILD_WRITER, cases=('J3',)),
    _JOBS + 'delete_for_entry': _child(Seam.CHILD_WRITER, cases=('J4',)),
    # The embedding CLI commands run outside any request and write by the ids they read.
    _CHUNKS + 'store_chunked': _child(
        Seam.CHILD_WRITER, _STORE_TX, _UPDATE_TX, 'cli/migrate_embeddings.py:_embed_missing_async', _REEMBED,
    ),
    _CHUNKS + '_store_chunked_compressed': _child(Seam.CHILD_WRITER, _CHUNK_WRITES + 'store_chunked'),
    _CHUNKS + 'delete_all_chunks': _child(
        Seam.CHILD_WRITER,
        _CHUNK_WRITES + 'store_chunked', _CHUNK_WRITES + 'delete_all_chunks_bulk', _UPDATE_TX, _UPDATE_TX, _REEMBED,
    ),
    _CHUNKS + '_delete_all_chunks_compressed': _child(
        Seam.CHILD_WRITER, _CHUNK_WRITES + '_store_chunked_compressed', _CHUNK_WRITES + 'delete_all_chunks',
    ),
    # The delete path hands over only the ids its in-transaction owner probe cleared.
    _CHUNKS + 'delete_all_chunks_bulk': _child(Seam.CHILD_WRITER, 'tools/_delete_cleanup.py:cleanup_embeddings_for_delete'),
    _CHUNKS + '_delete_all_chunks_compressed_bulk': _child(Seam.CHILD_WRITER, _CHUNK_WRITES + 'delete_all_chunks_bulk'),
    # Worker: the job queue and the generated outputs, written for whichever entry holds a job.
    _JOBS + 'claim_next': _worker(),
    _JOBS + 'renew_lease': _worker(),
    _JOBS + 'record_failure': _worker(receiver='generation_jobs'),
    _JOBS + 'delete_claimed': _worker(),
    _JOBS + 'delete_job': _worker(),
    _JOBS + 'release_claims': _worker(),
    _JOBS + 'reset_failed': _worker(),
    _GENERATION + 'read_generation_source': _worker(),
    _GENERATION + '_generation_source_sql': _worker(),
    _GENERATION + 'write_generation_summary': _worker(),
    # System: schema maintenance and deployment-level figures.
    _FTS_MAINTENANCE + 'is_available': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_current_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_desired_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'get_current_language': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'migrate_tokenizer': Entry(Seam.SYSTEM),
    _FTS_MAINTENANCE + 'migrate_language': Entry(Seam.SYSTEM),
    _INVENTORY + 'embedding_tables_exist': Entry(Seam.SYSTEM),
    _INVENTORY + 'get_embeddings_size': Entry(Seam.SYSTEM),
    _INVENTORY + '_get_embeddings_size_sqlite': Entry(Seam.SYSTEM),
}


@dataclass(frozen=True)
class ParsedModule:
    """One parsed module of ``app/`` and the ids of its docstring constants."""

    path: str  # relative to app/, POSIX separators
    dotted: str
    tree: ast.Module
    docstrings: frozenset[int]


type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _docstring_ids(tree: ast.Module) -> frozenset[int]:
    """Return the node ids of the module, class and function docstrings of ``tree``."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return frozenset(ids)


@functools.cache
def app_modules() -> tuple[ParsedModule, ...]:
    """Parse every module of ``app/`` once."""
    modules = []
    for file in sorted(APP.rglob('*.py')):
        parts = file.relative_to(ROOT).with_suffix('').parts
        dotted = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        tree = ast.parse(file.read_text(encoding='utf-8'))
        modules.append(ParsedModule(file.relative_to(APP).as_posix(), dotted, tree, _docstring_ids(tree)))
    return tuple(modules)


def string_constants(node: ast.AST, module: ParsedModule) -> Iterator[tuple[int, str]]:
    """Yield the line and value of every non-docstring string constant of ``node``'s subtree, f-string parts included."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and id(child) not in module.docstrings:
            yield child.lineno, child.value


def runs_context_sql(node: ast.AST, module: ParsedModule) -> bool:
    """Whether ``node``'s subtree carries a statement against a context table.

    A statement names its table in a string constant, or names a routable relation through
    the read seam.

    Returns:
        True when a string constant names a context table or the subtree calls the read seam.
    """
    if any(CONTEXT_TABLE_SQL.search(value) for _line, value in string_constants(node, module)):
        return True
    return any(isinstance(child, ast.Call) and callee_name(child) == READ_SEAM for child in ast.walk(node))


@functools.cache
def repository_methods() -> dict[str, tuple[FunctionNode, ParsedModule]]:
    """Map ``module:Class.method`` (or ``module:function``) to every method of ``app/repositories``."""
    methods: dict[str, tuple[FunctionNode, ParsedModule]] = {}
    for module in app_modules():
        if not module.path.startswith('repositories/'):
            continue
        for node in module.tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                methods[f'{module.dotted}:{node.name}'] = (node, module)
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                        methods[f'{module.dotted}:{node.name}.{member.name}'] = (member, module)
    return methods


def qualified_references(module: ParsedModule) -> Iterator[tuple[str, ast.Attribute]]:
    """Yield every attribute read of ``module`` with the qualified name of its enclosing function."""

    def visit(node: ast.AST, scope: tuple[str, ...]) -> Iterator[tuple[str, ast.Attribute]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                yield from visit(child, (*scope, child.name))
                continue
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                yield '.'.join(scope), child
            yield from visit(child, scope)

    yield from visit(module.tree, ())


def receiver_name(node: ast.expr) -> str | None:
    """The last name of the object an attribute is read from: ``repos.embeddings`` gives ``embeddings``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def method_name(key: str) -> str:
    """The bare name of the method a registry key names: ``module:Class.method`` gives ``method``."""
    return key.split(':', 1)[1].rsplit('.', 1)[-1]


def _within_worker_method(module: ParsedModule, function: str) -> bool:
    """Whether ``function``, qualified within ``module``, is a worker method or a closure inside one."""
    parts = function.split('.')
    candidates = {f'{module.dotted}:{".".join(parts[:2])}', f'{module.dotted}:{parts[0]}'}
    return any(key in REGISTRY and REGISTRY[key].seam is Seam.WORKER for key in candidates)


def worker_site_violations(modules: Iterable[ParsedModule]) -> list[str]:
    """List every reference to a worker method from code that may not call one.

    A worker method runs under whatever identity its caller passes and renders no access
    predicate, so only code that runs outside any request references it: the generation
    worker and the CLI (:data:`WORKER_CALLER_HOMES`), and other worker methods. A worker
    method nothing references yet passes.

    Args:
        modules: The parsed modules to search.

    Returns:
        ``<path under app>:<qualified function name>`` of every offending reference, sorted.
    """
    workers = {method_name(key): REGISTRY[key] for key in registry_keys(Seam.WORKER)}
    violations: list[str] = []
    for module in modules:
        if module.path.split('/')[0] in WORKER_CALLER_HOMES:
            continue
        for function, attribute in qualified_references(module):
            entry = workers.get(attribute.attr)
            if entry is None or (entry.receiver is not None and receiver_name(attribute.value) != entry.receiver):
                continue
            if not _within_worker_method(module, function):
                violations.append(f'{module.path}:{function}')
    return sorted(violations)


def callee_name(call: ast.Call) -> str | None:
    """The bare name a call invokes: ``f(...)`` and ``x.f(...)`` both give ``f``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def hands_over_scope(call: ast.Call) -> bool:
    """Whether a call passes the caller's own ``scope``, positionally or by keyword."""
    passed = [*call.args, *(keyword.value for keyword in call.keywords if keyword.arg == 'scope')]
    return any(isinstance(argument, ast.Name) and argument.id == 'scope' for argument in passed)


def direct_closures(node: ast.AST) -> dict[str, FunctionNode]:
    """Return the functions defined inside ``node`` but outside any function nested in it."""
    closures: dict[str, FunctionNode] = {}
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            closures[child.name] = child
        elif not isinstance(child, ast.Lambda | ast.ClassDef):
            closures.update(direct_closures(child))
    return closures


def resolve_target(target: str) -> tuple[ModuleType, Callable[..., object]]:
    """Import ``module:qualname`` and return the module and the function it names."""
    module_name, qualname = target.split(':')
    module = importlib.import_module(module_name)
    found: object = module
    for part in qualname.split('.'):
        found = getattr(found, part)
    assert callable(found), target
    return module, found


def _type_checking_names(module: ModuleType) -> dict[str, object]:
    """Import the names ``module`` binds only under ``if TYPE_CHECKING:``, for resolving its hints."""
    names: dict[str, object] = {}
    for node in ast.parse(inspect.getsource(module)).body:
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING'):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                source = importlib.import_module(statement.module)
                names.update({alias.asname or alias.name: getattr(source, alias.name) for alias in statement.names})
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    bound = alias.asname or alias.name.split('.')[0]
                    names[bound] = importlib.import_module(alias.name if alias.asname else bound)
    return names


def scope_parameter(target: str) -> tuple[inspect.Parameter, object]:
    """Return the ``scope`` parameter of ``target`` and its resolved annotation."""
    module, function = resolve_target(target)
    parameters = inspect.signature(function).parameters
    assert 'scope' in parameters, f'{target} takes no scope'
    hints = typing.get_type_hints(function, localns=_type_checking_names(module))
    return parameters['scope'], hints['scope']


def registry_keys(*seams: Seam) -> list[str]:
    """The registry keys of the given classes, sorted."""
    return sorted(key for key, entry in REGISTRY.items() if entry.seam in seams)

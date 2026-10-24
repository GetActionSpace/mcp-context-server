"""Regression tests for the CI job-skip conditions in .github/workflows.

Every quality gate (tests, pre-commit, MCP marker validation, Trivy security scan)
carries a job-level ``if:`` that exempts Release Please's own version-bump pull
request. The exemption must be authenticated by something the pull-request
submitter cannot control.

Two values were available. ``github.head_ref`` is the branch name, which a
contributor picks freely on a fork, so it is only safe when paired with a check that
the head branch lives in this repository: only an account with write access can push
a branch here, and a fork's branch never satisfies it. The pull-request author is
not that check, because the identity Release Please runs as is never the repository
owner once an organization owns the repository, and a literal login would tie the
exemption to one account. ``github.event.head_commit.message`` on the push-to-main event is worse:
depending on the repository's merge-message settings it embeds the fork branch
name, the pull-request title, or the squashed commit bodies, so it can be made to
contain any literal string a match looks for -- there is no way to authenticate the
release bot from it, and a matching push skipped pre-commit, the MCP checks and the
security scan (including its SARIF upload) on the default branch.

These tests are structural: they read the workflow files and pin the invariants, so
reintroducing an untrusted-string skip fails here rather than silently degrading the
signal a reviewer relies on.

Conditions are extracted with a real YAML parser rather than by matching lines. A
line-oriented reader has to re-derive scalar boundaries that YAML already defines,
and every shape it gets wrong makes a condition arrive TRUNCATED or EMPTY -- at
which point the substring assertions below still pass, on text that no longer
contains the thing they were written to reject. A multi-line plain scalar, a value
that begins on the line after the key, a block header carrying a trailing comment,
and a multi-line quoted scalar are all valid YAML and all defeat that approach.
PyYAML is a hard dependency of fastmcp, so parsing properly costs nothing here.
"""

from pathlib import Path

import pytest
import yaml

_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / '.github' / 'workflows'
# GitHub Actions reads BOTH extensions, so a workflow saved as .yaml would otherwise
# be exempt from every assertion in this module while still gating the repository.
_WORKFLOW_FILES = sorted(
    path for pattern in ('*.yml', '*.yaml') for path in _WORKFLOW_DIR.glob(pattern)
)


def _collect_if_values(node: object, found: list[str]) -> None:
    """Append every ``if`` value found anywhere in a parsed workflow document.

    Args:
        node: A node of the parsed YAML document.
        found: Accumulator receiving one entry per ``if`` key, in document order.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'if':
                found.append(str(value))
            else:
                _collect_if_values(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_if_values(item, found)


def _conditions(path: Path) -> list[str]:
    """Return every job/step ``if:`` expression in one workflow file, one per ``if:`` key.

    Args:
        path: Workflow file to read.

    Returns:
        The text of each condition, in document order, exactly one entry per ``if:`` key.
    """
    found: list[str] = []
    _collect_if_values(yaml.safe_load(path.read_text(encoding='utf-8')), found)
    return found


def test_workflow_files_are_present() -> None:
    """Guard against the glob silently matching nothing."""
    assert _WORKFLOW_FILES, f'no workflow files found under {_WORKFLOW_DIR}'


@pytest.mark.parametrize('workflow', _WORKFLOW_FILES, ids=lambda p: p.name)
def test_no_job_gates_on_the_commit_message(workflow: Path) -> None:
    """No job may decide whether to run by matching the pushed commit message."""
    text = workflow.read_text(encoding='utf-8')
    assert 'head_commit.message' not in text, (
        f'{workflow.name} gates on the commit message, which a contributor can control '
        f'through the fork branch name, the pull-request title or a squashed commit body'
    )


@pytest.mark.parametrize('workflow', _WORKFLOW_FILES, ids=lambda p: p.name)
def test_branch_prefix_skips_are_paired_with_a_same_repository_check(workflow: Path) -> None:
    """A skip keyed on the branch name must also require the branch to live in this repository."""
    for condition in _conditions(workflow):
        if 'head_ref' not in condition:
            continue
        assert 'github.event.pull_request.head.repo.full_name' in condition, (
            f'{workflow.name} skips on the head branch name alone; a contributor names a '
            f'fork branch freely, so the exemption must also require the head branch to '
            f'live in this repository, which only an account with write access can push'
        )
        assert 'github.repository' in condition, (
            f'{workflow.name} compares the head repository against something other than '
            f'this repository, which is the only one the release bot pushes its branch to'
        )


def test_the_quality_gates_still_carry_a_skip_condition() -> None:
    """The two gate workflows must keep an authenticated exemption, not lose it entirely.

    Without this the previous test passes vacuously if every condition is deleted.
    """
    for name in ('lint.yml', 'test.yml'):
        path = _WORKFLOW_DIR / name
        assert path.exists(), f'{name} is missing'
        conditions = [c for c in _conditions(path) if 'head_ref' in c]
        assert conditions, f'{name} no longer exempts the release bump pull request'


def test_every_condition_arrives_whole_and_separate(tmp_path: Path) -> None:
    """Each ``if:`` yields exactly one complete condition, whatever scalar shape it uses.

    Covers the shapes a line-oriented reader mis-handles, each of which silently
    empties or truncates a condition and so lets an unguarded branch-prefix skip pass
    the same-repository assertion above: a multi-line plain scalar, a value starting on
    the line after the key, a block header followed by a comment, and a multi-line
    double-quoted scalar. It also pins the separation between an unwrapped condition
    and a neighbouring job's wrapped one, which a scan terminating at the next ``}}``
    merged into a single blob carrying both jobs' substrings.
    """
    workflow = tmp_path / 'shapes.yml'
    workflow.write_text(
        'jobs:\n'
        '  multiline_plain:\n'
        '    if: always()\n'
        "      && startsWith(github.head_ref, 'release-please--branches--')\n"
        '    steps:\n'
        '      - run: echo a\n'
        '  value_on_next_line:\n'
        '    if:\n'
        "      startsWith(github.head_ref, 'release-please--branches--')\n"
        '    steps:\n'
        '      - run: echo b\n'
        '  block_header_with_comment:\n'
        '    if: >- # keep the branch check on its own line\n'
        "      startsWith(github.head_ref, 'release-please--branches--')\n"
        '    steps:\n'
        '      - run: echo c\n'
        '  multiline_quoted:\n'
        '    if: "always()\n'
        '      && startsWith(github.head_ref, \'release-please--branches--\')"\n'
        '    steps:\n'
        '      - run: echo d\n'
        '  wrapped:\n'
        '    if: ${{ github.event.pull_request.head.repo.full_name == github.repository }}\n'
        '    steps:\n'
        '      - run: echo e\n',
        encoding='utf-8',
    )

    conditions = _conditions(workflow)

    assert len(conditions) == 5
    for condition in conditions[:4]:
        assert 'head_ref' in condition, condition
        assert 'head.repo.full_name' not in condition, condition
    assert 'head.repo.full_name' in conditions[4]
    assert 'head_ref' not in conditions[4]


def test_step_level_conditions_are_collected_too(tmp_path: Path) -> None:
    """A skip placed on a STEP is inspected the same way a job-level one is.

    A gate can be disabled just as effectively one level down, so the extraction must
    reach into the steps list rather than stopping at the job mapping.
    """
    workflow = tmp_path / 'step_level.yml'
    workflow.write_text(
        'jobs:\n'
        '  build:\n'
        '    steps:\n'
        '      - run: echo guarded\n'
        "        if: startsWith(github.head_ref, 'release-please--branches--')\n",
        encoding='utf-8',
    )

    conditions = _conditions(workflow)

    assert len(conditions) == 1
    assert 'head_ref' in conditions[0]

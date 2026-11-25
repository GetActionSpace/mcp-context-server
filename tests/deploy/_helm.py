"""Shared helpers for the tests that render the Helm chart with ``helm template``.

``HELM`` is None where the ``helm`` binary is unavailable; a test module that uses these
helpers skips itself in that case.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

HELM = shutil.which('helm')
ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / 'deploy' / 'helm' / 'mcp-context-server'
SQLITE_VALUES = ('-f', str(CHART / 'values-sqlite.yaml'))
POSTGRESQL_VALUES = ('-f', str(CHART / 'values-postgresql.yaml'), '--set', 'storage.postgresql.password=p')


def helm_template(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``helm template`` on the chart.

    Args:
        *args: Extra ``helm template`` arguments.

    Returns:
        The completed process.
    """
    assert HELM is not None
    return subprocess.run(
        [HELM, 'template', 't', str(CHART), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def render(*args: str) -> list[dict[str, Any]]:
    """Render the chart and parse every manifest.

    Args:
        *args: Extra ``helm template`` arguments.

    Returns:
        The rendered manifests.
    """
    result = helm_template(*args)
    assert result.returncode == 0, f'helm template failed: {result.stderr}'
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def manifest(docs: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    """Return the single manifest of one kind, if rendered.

    Args:
        docs: The rendered manifests.
        kind: The Kubernetes kind.

    Returns:
        The manifest, or None when the chart renders none of that kind.
    """
    found = [doc for doc in docs if doc.get('kind') == kind]
    assert len(found) <= 1, f'more than one {kind}'
    return found[0] if found else None


def configmap_data(docs: list[dict[str, Any]]) -> dict[str, str]:
    """Return the ConfigMap data.

    Args:
        docs: The rendered manifests.

    Returns:
        The ConfigMap data mapping.
    """
    configmap = manifest(docs, 'ConfigMap')
    assert configmap is not None
    data = configmap['data']
    assert isinstance(data, dict)
    return data

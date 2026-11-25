"""Helm-template tests for the HTTP keep-alive timeout the chart hands to the server.

The ConfigMap carries ``HTTP_KEEP_ALIVE_TIMEOUT_S`` from ``env.HTTP_KEEP_ALIVE_TIMEOUT_S``,
so the chart's default and an override in the values both reach the server.

These render the chart with ``helm template`` and parse the manifests, so they
require the ``helm`` binary and are skipped where it is unavailable.
"""

import pytest

from tests.deploy._helm import HELM
from tests.deploy._helm import POSTGRESQL_VALUES
from tests.deploy._helm import SQLITE_VALUES
from tests.deploy._helm import configmap_data
from tests.deploy._helm import render

pytestmark = pytest.mark.skipif(HELM is None, reason='helm binary not available')


@pytest.mark.parametrize(
    'profile',
    [(), SQLITE_VALUES, POSTGRESQL_VALUES],
    ids=['default', 'sqlite', 'postgresql'],
)
def test_configmap_carries_the_default_keep_alive_timeout(profile: tuple[str, ...]) -> None:
    """The default, SQLite and PostgreSQL values render the 75-second default into the ConfigMap."""
    assert configmap_data(render(*profile))['HTTP_KEEP_ALIVE_TIMEOUT_S'] == '75'


def test_configmap_carries_an_overridden_keep_alive_timeout() -> None:
    """A value set for env.HTTP_KEEP_ALIVE_TIMEOUT_S renders into the ConfigMap."""
    assert configmap_data(render('--set', 'env.HTTP_KEEP_ALIVE_TIMEOUT_S=120'))['HTTP_KEEP_ALIVE_TIMEOUT_S'] == '120'

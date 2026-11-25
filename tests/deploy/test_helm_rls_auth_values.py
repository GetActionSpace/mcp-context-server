"""Helm-template tests for the row-level security, authentication and access-control values.

The chart renders each of these server variables only when its value is set, so an
untouched value keeps the server default and nothing drifts from ``app/settings``.
Plain values go to the ConfigMap; the migration password, the bearer token and the
JWT public key go to the chart Secret or come from an existing secret, following the
``postgresql-password`` pattern. Enabling row-level security without a migration
user, or with the SQLite backend, fails at template time, mirroring the settings
validators.

These render the chart with ``helm template`` and parse the manifests, so they
require the ``helm`` binary and are skipped where it is unavailable.
"""

import base64
import json
from typing import Any

import pytest

from tests.deploy._helm import HELM
from tests.deploy._helm import POSTGRESQL_VALUES
from tests.deploy._helm import ROOT
from tests.deploy._helm import SQLITE_VALUES
from tests.deploy._helm import configmap_data
from tests.deploy._helm import helm_template
from tests.deploy._helm import manifest
from tests.deploy._helm import render

pytestmark = pytest.mark.skipif(HELM is None, reason='helm binary not available')

_FULLNAME = 't-mcp-context-server'

_CONFIG_KEYS = (
    'ENABLE_POSTGRESQL_RLS',
    'POSTGRESQL_RLS_POLICY_ROLE',
    'POSTGRESQL_MIGRATION_USER',
    'MCP_AUTH_PROVIDER',
    'MCP_AUTH_CLIENT_ID',
    'MCP_AUTH_JWT_JWKS_URI',
    'MCP_AUTH_JWT_ISSUER',
    'MCP_AUTH_JWT_AUDIENCE',
    'MCP_AUTH_JWT_ALGORITHM',
    'MCP_AUTH_ANONYMOUS_JWT_ALGORITHM',
    'MCP_AUTH_GROUPS_CLAIM',
    'MCP_AUTH_ROLES_CLAIM',
    'ACCESS_CONTROL_DEFAULT_PRINCIPAL',
    'ACCESS_CONTROL_DEFAULT_VISIBILITY',
    'ACCESS_CONTROL_DEFAULT_GROUP_GRANTS',
    'ACCESS_CONTROL_PUBLISH_ROLE',
    'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL',
)
_SECRET_ENVS = (
    'POSTGRESQL_MIGRATION_PASSWORD',
    'MCP_AUTH_TOKEN',
    'MCP_AUTH_JWT_PUBLIC_KEY',
    'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY',
)
_SECRET_KEYS = (
    'postgresql-migration-password',
    'mcp-auth-token',
    'mcp-auth-jwt-public-key',
    'mcp-auth-anonymous-jwt-public-key',
)
_JWKS = ('--set', 'auth.provider=jwt', '--set', 'auth.jwt.jwksUri=https://idp.example/jwks.json')


def _secret_data(docs: list[dict[str, Any]]) -> dict[str, str]:
    """Return the decoded chart Secret data, empty when the chart renders no Secret.

    Args:
        docs: The rendered manifests.

    Returns:
        The Secret data with each value base64-decoded.
    """
    secret = manifest(docs, 'Secret')
    if secret is None:
        return {}
    return {key: base64.b64decode(value).decode() for key, value in secret['data'].items()}


def _env(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the server container's env entries by name, rejecting duplicates.

    Args:
        docs: The rendered manifests.

    Returns:
        Each env entry keyed by its name.
    """
    deployment = manifest(docs, 'Deployment')
    assert deployment is not None
    containers = deployment['spec']['template']['spec']['containers']
    server = next(container for container in containers if container['name'] == 'mcp-context-server')
    entries = server.get('env') or []
    names = [entry['name'] for entry in entries]
    assert len(names) == len(set(names)), f'duplicate env entries: {names}'
    return {entry['name']: entry for entry in entries}


def _secret_ref(entry: dict[str, Any]) -> tuple[str, str]:
    """Return the secret name and key an env entry reads.

    Args:
        entry: One container env entry.

    Returns:
        The secret name and the data key.
    """
    ref = entry['valueFrom']['secretKeyRef']
    return ref['name'], ref['key']


@pytest.mark.parametrize(
    'profile',
    [(), SQLITE_VALUES, POSTGRESQL_VALUES],
    ids=['default', 'sqlite', 'postgresql'],
)
def test_untouched_values_render_none_of_the_variables(profile: tuple[str, ...]) -> None:
    """With every new value left empty the server defaults apply: nothing is rendered."""
    docs = render(*profile)
    config = configmap_data(docs)
    env = _env(docs)
    secret = _secret_data(docs)
    for key in _CONFIG_KEYS:
        assert key not in config, key
    for name in _SECRET_ENVS:
        assert name not in env, name
    for key in _SECRET_KEYS:
        assert key not in secret, key


def test_rls_values_render_the_toggle_policy_role_and_migration_credentials() -> None:
    """Row-level security renders its toggle, the policy role and the migration user and password."""
    docs = render(
        *POSTGRESQL_VALUES,
        '--set', 'storage.postgresql.user=mcp_app',
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.migration.password=owner-secret',
        '--set', 'storage.postgresql.rls.enabled=true',
        '--set', 'storage.postgresql.rls.policyRole=mcp_ctx',
    )
    config = configmap_data(docs)
    assert config['ENABLE_POSTGRESQL_RLS'] == 'true'
    assert config['POSTGRESQL_RLS_POLICY_ROLE'] == 'mcp_ctx'
    assert config['POSTGRESQL_MIGRATION_USER'] == 'owner'
    assert config['POSTGRESQL_USER'] == 'mcp_app'
    assert _secret_data(docs)['postgresql-migration-password'] == 'owner-secret'
    assert _secret_ref(_env(docs)['POSTGRESQL_MIGRATION_PASSWORD']) == (_FULLNAME, 'postgresql-migration-password')


def test_rls_without_a_policy_role_keeps_the_server_default() -> None:
    """An empty policy role renders no variable, so the server default role applies."""
    config = configmap_data(render(
        *POSTGRESQL_VALUES,
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.rls.enabled=true',
    ))
    assert config['ENABLE_POSTGRESQL_RLS'] == 'true'
    assert 'POSTGRESQL_RLS_POLICY_ROLE' not in config


def test_migration_user_alone_renders_without_rls() -> None:
    """The migration credentials render on their own, with row-level security left off."""
    docs = render(
        *POSTGRESQL_VALUES,
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.migration.password=owner-secret',
    )
    config = configmap_data(docs)
    assert config['POSTGRESQL_MIGRATION_USER'] == 'owner'
    assert 'ENABLE_POSTGRESQL_RLS' not in config
    assert 'POSTGRESQL_MIGRATION_PASSWORD' in _env(docs)


def test_migration_values_are_ignored_without_postgresql() -> None:
    """With the SQLite backend the migration credentials render nothing."""
    docs = render(
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.migration.password=owner-secret',
    )
    assert 'POSTGRESQL_MIGRATION_USER' not in configmap_data(docs)
    assert 'POSTGRESQL_MIGRATION_PASSWORD' not in _env(docs)
    assert 'postgresql-migration-password' not in _secret_data(docs)


@pytest.mark.parametrize(
    ('key_override', 'expected_key'),
    [
        ((), 'postgresql-migration-password'),
        (('--set', 'storage.postgresql.migration.existingSecretKey=owner-pw'), 'owner-pw'),
    ],
    ids=['default-key', 'key-override'],
)
def test_migration_password_from_an_existing_secret(key_override: tuple[str, ...], expected_key: str) -> None:
    """An existing secret supplies the migration password and the chart Secret carries none."""
    docs = render(
        *POSTGRESQL_VALUES,
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.migration.password=ignored',
        '--set', 'storage.postgresql.migration.existingSecret=migsec',
        *key_override,
    )
    assert _secret_ref(_env(docs)['POSTGRESQL_MIGRATION_PASSWORD']) == ('migsec', expected_key)
    assert 'postgresql-migration-password' not in _secret_data(docs)


def test_rls_without_a_migration_user_fails() -> None:
    """Row-level security needs the migration role that owns the tables."""
    result = helm_template(*POSTGRESQL_VALUES, '--set', 'storage.postgresql.rls.enabled=true')
    assert result.returncode != 0
    assert 'storage.postgresql.migration.user' in result.stderr
    assert 'POSTGRESQL_MIGRATION_USER' in result.stderr


def test_rls_with_the_sqlite_backend_fails() -> None:
    """Row-level security is a PostgreSQL feature."""
    result = helm_template(
        '--set', 'storage.postgresql.migration.user=owner',
        '--set', 'storage.postgresql.rls.enabled=true',
    )
    assert result.returncode != 0
    assert 'ENABLE_POSTGRESQL_RLS requires STORAGE_BACKEND=postgresql' in result.stderr


def test_auth_and_access_control_values_render_in_the_configmap() -> None:
    """Every plain authentication and access-control value renders under its server variable."""
    config = configmap_data(render(
        '--set', 'auth.provider=jwt',
        '--set', 'auth.clientId=chart-client',
        '--set', 'auth.jwt.jwksUri=https://idp.example/jwks.json',
        '--set', 'auth.jwt.issuer=https://idp.example/',
        '--set', 'auth.jwt.audience=mcp-context',
        '--set', 'auth.jwt.algorithm=ES256',
        '--set', 'auth.jwt.anonymousAlgorithm=ES384',
        '--set', 'auth.groupsClaim=realm_access.groups',
        '--set', 'auth.rolesClaim=realm_access.roles',
        '--set', 'accessControl.defaultPrincipal=service',
        '--set', 'accessControl.defaultVisibility=public',
        '--set', 'accessControl.defaultGroupGrants=author_groups',
        '--set', 'accessControl.publishRole=publisher',
        '--set', 'accessControl.anonymousPrincipal=anonymous',
    ))
    assert {key: config[key] for key in config if key.startswith(('MCP_AUTH_', 'ACCESS_CONTROL_'))} == {
        'MCP_AUTH_PROVIDER': 'jwt',
        'MCP_AUTH_CLIENT_ID': 'chart-client',
        'MCP_AUTH_JWT_JWKS_URI': 'https://idp.example/jwks.json',
        'MCP_AUTH_JWT_ISSUER': 'https://idp.example/',
        'MCP_AUTH_JWT_AUDIENCE': 'mcp-context',
        'MCP_AUTH_JWT_ALGORITHM': 'ES256',
        'MCP_AUTH_ANONYMOUS_JWT_ALGORITHM': 'ES384',
        'MCP_AUTH_GROUPS_CLAIM': 'realm_access.groups',
        'MCP_AUTH_ROLES_CLAIM': 'realm_access.roles',
        'ACCESS_CONTROL_DEFAULT_PRINCIPAL': 'service',
        'ACCESS_CONTROL_DEFAULT_VISIBILITY': 'public',
        'ACCESS_CONTROL_DEFAULT_GROUP_GRANTS': 'author_groups',
        'ACCESS_CONTROL_PUBLISH_ROLE': 'publisher',
        'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL': 'anonymous',
    }


def test_inline_token_and_public_key_go_to_the_chart_secret() -> None:
    """Inline secrets land in the chart Secret and the container reads them from it."""
    docs = render(
        '--set', 'auth.provider=simple_token',
        '--set', 'auth.token=bearer-value',
        '--set', 'auth.jwt.publicKey=hs-shared-secret',
    )
    secret = _secret_data(docs)
    assert secret['mcp-auth-token'] == 'bearer-value'
    assert secret['mcp-auth-jwt-public-key'] == 'hs-shared-secret'
    env = _env(docs)
    assert _secret_ref(env['MCP_AUTH_TOKEN']) == (_FULLNAME, 'mcp-auth-token')
    assert _secret_ref(env['MCP_AUTH_JWT_PUBLIC_KEY']) == (_FULLNAME, 'mcp-auth-jwt-public-key')


@pytest.mark.parametrize(
    ('key_override', 'expected_key'),
    [((), 'mcp-auth-token'), (('--set', 'auth.tokenSecretKey=bearer'), 'bearer')],
    ids=['default-key', 'key-override'],
)
def test_token_from_an_existing_secret_with_simple_token(key_override: tuple[str, ...], expected_key: str) -> None:
    """simple_token reads its token from the existing secret and references no public key."""
    docs = render(
        '--set', 'auth.provider=simple_token',
        '--set', 'auth.token=ignored',
        '--set', 'auth.existingSecret=authsec',
        *key_override,
    )
    env = _env(docs)
    assert _secret_ref(env['MCP_AUTH_TOKEN']) == ('authsec', expected_key)
    assert 'MCP_AUTH_JWT_PUBLIC_KEY' not in env
    assert 'mcp-auth-token' not in _secret_data(docs)


@pytest.mark.parametrize(
    ('key_override', 'expected_key'),
    [((), 'mcp-auth-jwt-public-key'), (('--set', 'auth.jwt.publicKeySecretKey=jwt-pem'), 'jwt-pem')],
    ids=['default-key', 'key-override'],
)
def test_public_key_from_an_existing_secret_with_jwt(key_override: tuple[str, ...], expected_key: str) -> None:
    """jwt without a JWKS URI reads its public key from the existing secret and references no token."""
    docs = render(
        '--set', 'auth.provider=jwt',
        '--set', 'auth.existingSecret=authsec',
        *key_override,
    )
    env = _env(docs)
    assert _secret_ref(env['MCP_AUTH_JWT_PUBLIC_KEY']) == ('authsec', expected_key)
    assert 'MCP_AUTH_TOKEN' not in env
    assert 'mcp-auth-jwt-public-key' not in _secret_data(docs)


def test_jwks_uri_with_an_existing_secret_references_no_public_key() -> None:
    """jwt with a JWKS URI needs no public key, so the existing secret is not referenced for one."""
    env = _env(render(
        '--set', 'auth.provider=jwt',
        '--set', 'auth.existingSecret=authsec',
        '--set', 'auth.jwt.jwksUri=https://idp.example/jwks.json',
    ))
    assert 'MCP_AUTH_JWT_PUBLIC_KEY' not in env
    assert 'MCP_AUTH_TOKEN' not in env


def test_inline_anonymous_key_goes_to_the_chart_secret_on_a_jwks_deployment() -> None:
    """With jwt and a JWKS URI, the inline anonymous key lands in the chart Secret and the container reads it."""
    docs = render(*_JWKS, '--set', 'auth.jwt.anonymousPublicKey=anonymous-pem')
    assert _secret_data(docs)['mcp-auth-anonymous-jwt-public-key'] == 'anonymous-pem'
    env = _env(docs)
    assert _secret_ref(env['MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY']) == (_FULLNAME, 'mcp-auth-anonymous-jwt-public-key')
    assert 'MCP_AUTH_JWT_PUBLIC_KEY' not in env


@pytest.mark.parametrize(
    'deployment',
    [
        ('--set', 'auth.provider=jwt', '--set', 'auth.jwt.publicKey=static-pem'),
        ('--set', 'auth.provider=simple_token', '--set', 'auth.jwt.jwksUri=https://idp.example/jwks.json'),
        ('--set', 'auth.jwt.jwksUri=https://idp.example/jwks.json'),
    ],
    ids=['jwt-static-key', 'simple-token', 'no-provider'],
)
def test_inline_anonymous_key_renders_only_for_jwt_with_a_jwks_uri(deployment: tuple[str, ...]) -> None:
    """The anonymous verifier exists only beside a JWKS verifier, so no other deployment renders its key."""
    docs = render(*deployment, '--set', 'auth.jwt.anonymousPublicKey=anonymous-pem')
    assert 'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY' not in _env(docs)
    assert 'mcp-auth-anonymous-jwt-public-key' not in _secret_data(docs)


@pytest.mark.parametrize(
    ('key_override', 'expected_key'),
    [
        ((), 'mcp-auth-anonymous-jwt-public-key'),
        (('--set', 'auth.jwt.anonymousPublicKeySecretKey=anon-pem'), 'anon-pem'),
    ],
    ids=['default-key', 'key-override'],
)
def test_anonymous_key_from_an_existing_secret_on_a_jwks_deployment(
    key_override: tuple[str, ...], expected_key: str,
) -> None:
    """An existing secret supplies the anonymous key on a JWKS deployment, as an optional reference.

    A JWKS deployment whose secret carries no anonymous key keeps starting: the
    reference is optional, and without the variable the server composes no
    anonymous verifier.
    """
    docs = render(
        *_JWKS,
        '--set', 'auth.existingSecret=authsec',
        '--set', 'auth.jwt.anonymousPublicKey=ignored',
        *key_override,
    )
    entry = _env(docs)['MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY']
    assert _secret_ref(entry) == ('authsec', expected_key)
    assert entry['valueFrom']['secretKeyRef']['optional'] is True
    assert 'mcp-auth-anonymous-jwt-public-key' not in _secret_data(docs)


def test_existing_secret_without_a_jwks_uri_references_no_anonymous_key() -> None:
    """A static-key jwt deployment reads only its public key from the existing secret."""
    env = _env(render('--set', 'auth.provider=jwt', '--set', 'auth.existingSecret=authsec'))
    assert 'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY' not in env
    assert 'MCP_AUTH_JWT_PUBLIC_KEY' in env


def test_every_rendered_variable_is_a_server_setting() -> None:
    """Each variable the chart renders for these values is an environment variable the server reads."""
    server_json = json.loads((ROOT / 'server.json').read_text(encoding='utf-8'))
    known = {
        variable['name']
        for package in server_json['packages']
        for variable in package.get('environmentVariables', [])
    }
    missing = sorted(set(_CONFIG_KEYS + _SECRET_ENVS) - known)
    assert not missing, f'not server settings: {missing}'

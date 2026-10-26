# Authentication Guide

## Introduction

This guide covers authentication options for the MCP Context Server when using HTTP transports. Authentication is handled via a configurable provider, with bearer token authentication available for HTTP deployments.

**Key Concepts:**
- Authentication is **only relevant for HTTP transports** (http, sse, streamable-http)
- STDIO transport (default) provides process-level security without authentication
- Three authentication modes available: no auth (STDIO), bearer token (HTTP), and JWT verification (HTTP)
- Every request reads and writes as one principal: the token's subject under JWT verification, and the configured default principal otherwise (see [Access Model](#access-model))
- Configuration is entirely via environment variables

## Authentication Methods Overview

| Method            | Transport | Use Case                                             |
|-------------------|-----------|------------------------------------------------------|
| No Authentication | STDIO     | Local development, Claude Desktop, CLI tools         |
| Bearer Token      | HTTP      | Simple API access, CI/CD, internal services          |
| JWT Verification  | HTTP      | Tokens issued by an identity provider (OIDC/OAuth2)  |

## No Authentication (STDIO)

### When to Use

- Claude Desktop and Claude Code CLI (default configuration)
- Local development and testing
- Single-user deployments
- Trusted network environments

### How It Works

When using STDIO transport (`MCP_TRANSPORT=stdio`, which is the default), the MCP server runs as a subprocess spawned by the client. Security is provided at the process level:

1. Client spawns server as a child process
2. Communication occurs via stdin/stdout
3. No network exposure
4. OS-level process isolation

### Configuration

No authentication configuration needed. This is the default behavior:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-context-server"]
    }
  }
}
```

### Security Considerations

- Server only accessible to the parent process
- No network ports exposed
- File system permissions determine database access
- Suitable for personal/development use

## Bearer Token Authentication

### When to Use

- HTTP transport deployments requiring simple authentication
- CI/CD pipelines and automation
- Internal microservices communication
- Docker deployments with controlled access

### How It Works

The `SimpleTokenVerifier` class validates bearer tokens against a static token configured via environment variables. Key security features:

- **SecretStr handling**: Token never exposed in logs or error messages
- **Constant-time comparison**: Prevents timing attacks via `hmac.compare_digest()`
- **Centralized configuration**: Uses `AuthSettings` for consistent settings management

The bearer token proves access to the server but names no principal. Every `simple_token` request reads and writes as `ACCESS_CONTROL_DEFAULT_PRINCIPAL` with no groups or roles, exactly like STDIO and `MCP_AUTH_PROVIDER=none`, so all token holders share one owner identity. `MCP_AUTH_CLIENT_ID` identifies the client to FastMCP and is not an owner identity. A `simple_token` caller cannot reach entries owned by a client id such as `mcp-client`: it cannot read the private ones and cannot change or delete any of them. Hand them to the default principal with `mcp-context-server-migrate --source-url <url> --reassign-owner mcp-client local`, replacing `local` with your `ACCESS_CONTROL_DEFAULT_PRINCIPAL` value if you changed it (see [Reassigning Entry Ownership](migration-v2-to-v3.md#reassigning-entry-ownership)).

### Configuration

**Required Environment Variables:**

| Variable             | Required | Description                                                                                    |
|----------------------|----------|------------------------------------------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`  | Yes      | Set to `simple_token`                                                                          |
| `MCP_AUTH_TOKEN`     | Yes      | The bearer token for authentication                                                            |
| `MCP_AUTH_CLIENT_ID` | No       | Client ID that identifies the client to FastMCP; not an owner identity (default: `mcp-client`) |

**Example Configuration:**

```bash
# .env file or environment variables
MCP_TRANSPORT=http
FASTMCP_HOST=0.0.0.0
FASTMCP_PORT=8000
MCP_AUTH_PROVIDER=simple_token
MCP_AUTH_TOKEN=your-secret-token-here
MCP_AUTH_CLIENT_ID=my-service
```

### Client Configuration

Clients must include the bearer token in the `Authorization` header:

```text
Authorization: Bearer your-secret-token-here
```

**Claude Code CLI:**

```bash
# Add HTTP server with Bearer token authentication
claude mcp add --transport http context-server http://localhost:8000/mcp --header "Authorization: Bearer your-secret-token-here"
```

**HTTP Client Example (curl):**

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

**Python Client Example:**

```python
import httpx

headers = {
    "Authorization": "Bearer your-secret-token-here",
    "Content-Type": "application/json"
}

response = httpx.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
)
```

### Security Best Practices

1. **Use strong tokens**: Generate cryptographically secure tokens (32+ characters)
   ```bash
   # Generate secure token
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Never commit tokens**: Use environment variables or secrets management

3. **Use HTTPS in production**: Token transmitted in header requires TLS

4. **Rotate tokens regularly**: Change tokens periodically for long-running deployments

## JWT Authentication

The `jwt` provider verifies IdP-issued tokens, rejects unauthenticated requests, and runs every request as the token's principal: each stored entry is owned by the principal that wrote it, every read, update and delete reaches only the entries that principal may access, an entry's owner shares it with users and groups through `share_context`, and every returned entry reports its `visibility` and whether the caller owns it (see [Access Model](#access-model), [Sharing Entries](#sharing-entries) and [Anonymous Read Access](#anonymous-read-access)). Prefer PostgreSQL with row-level security for deployments with more than one principal (see [Enforcement and Its Limits](#enforcement-and-its-limits)).

### When to Use

- Tokens are issued by an identity provider (Keycloak, Microsoft Entra ID, Auth0, or any OIDC/OAuth2-compliant issuer)
- Multiple clients or users need individually issued, expiring credentials
- Key rotation should happen at the IdP without server reconfiguration (JWKS mode)

### How It Works

With `MCP_AUTH_PROVIDER=jwt`, the server validates each bearer token as a JWT: its signature (static key or JWKS lookup), its `exp` expiry, and, when configured, its issuer and audience. A token without an `exp` claim never expires, except that the server refuses such a token for the [anonymous principal](#anonymous-read-access), and the `nbf` (not before) claim is not checked. Verified claims from the token (subject, groups, roles) are available to the server for identity resolution.

Two key modes are supported, and exactly one must be configured:

- **JWKS mode** (`MCP_AUTH_JWT_JWKS_URI`): the server fetches signing keys from the IdP's JWKS endpoint and caches them, so IdP-side key rotation needs no server change. This is the recommended mode for real IdPs. The HTTPS connection to the JWKS endpoint is verified against the operating system's certificate store; for an IdP behind a private CA, add the CA to that store or point `SSL_CERT_FILE`/`SSL_CERT_DIR` at it.
- **Static key mode** (`MCP_AUTH_JWT_PUBLIC_KEY`): a fixed PEM-encoded public key (asymmetric algorithms) or shared secret (HS* algorithms). Suited to testing and closed environments without a reachable JWKS endpoint.

### Configuration

| Variable                  | Required             | Description                                                                          |
|---------------------------|----------------------|--------------------------------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`       | Yes                  | Set to `jwt`                                                                         |
| `MCP_AUTH_JWT_JWKS_URI`   | One of the key pair  | JWKS endpoint URI (mutually exclusive with the public key)                           |
| `MCP_AUTH_JWT_PUBLIC_KEY` | One of the key pair  | PEM public key or HS* shared secret (mutually exclusive with the JWKS URI)           |
| `MCP_AUTH_JWT_ISSUER`     | Recommended          | Expected `iss` claim; unset skips issuer validation                                  |
| `MCP_AUTH_JWT_AUDIENCE`   | Recommended          | Expected `aud` claim; unset skips audience validation                                |
| `MCP_AUTH_JWT_ALGORITHM`  | No (default `RS256`) | Accepted signing algorithm                                                           |
| `MCP_AUTH_GROUPS_CLAIM`   | No (default `groups`)| Claim carrying group memberships; dotted paths and full-URL keys supported           |
| `MCP_AUTH_ROLES_CLAIM`    | No (default `roles`) | Claim carrying roles; dotted paths and full-URL keys supported                       |

**Example Configuration (JWKS mode):**

```bash
MCP_TRANSPORT=http
MCP_AUTH_PROVIDER=jwt
MCP_AUTH_JWT_JWKS_URI=https://idp.example.com/realms/main/protocol/openid-connect/certs
MCP_AUTH_JWT_ISSUER=https://idp.example.com/realms/main
MCP_AUTH_JWT_AUDIENCE=mcp-context-server
```

Clients send the IdP-issued token exactly like a bearer token: `Authorization: Bearer <jwt>`.

### Claim Mapping

Group and role claims differ across identity providers, so the claim keys are configurable:

- A plain key (`groups`) reads the top-level claim. RFC 9068 names `groups` as the standard claim, which is why it is the default.
- A dotted key (`realm_access.roles`) first tries the literal flat key, then traverses nested claim objects.
- A full-URL key (`https://example.com/groups`) is looked up verbatim; the dots inside the URL are never treated as a path.
- A scalar string claim value is treated as a single-element list.

### Per-IdP Setup

#### Keycloak

Keycloak realm roles arrive nested under `realm_access.roles` automatically; set `MCP_AUTH_ROLES_CLAIM=realm_access.roles` to read them. Group membership requires a mapper: in the client's dedicated scope, add a **Group Membership** mapper with token claim name `groups` (disable **Full group path** unless you want `/parent/child` names). The defaults then work unchanged.

```bash
MCP_AUTH_JWT_JWKS_URI=https://keycloak.example.com/realms/<realm>/protocol/openid-connect/certs
MCP_AUTH_JWT_ISSUER=https://keycloak.example.com/realms/<realm>
MCP_AUTH_JWT_AUDIENCE=<client-id>
MCP_AUTH_ROLES_CLAIM=realm_access.roles
```

#### Microsoft Entra ID

Enable group claims in the app registration (**Token configuration** > **Add groups claim**, or `groupMembershipClaims` in the manifest). Entra emits group **object IDs (GUIDs)**, not display names -- grants and policies must use those GUIDs. When a user belongs to more than the token limit (about 200 groups for JWTs), Entra omits the `groups` claim and emits `_claim_names`/`_claim_sources` overage markers instead; this server treats that case as an EMPTY group list (fail-closed) and logs a warning -- it never calls Microsoft Graph to resolve the membership. Avoid overage by filtering to groups assigned to the application in the app registration.

```bash
MCP_AUTH_JWT_JWKS_URI=https://login.microsoftonline.com/<tenant-id>/discovery/v2.0/keys
MCP_AUTH_JWT_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
MCP_AUTH_JWT_AUDIENCE=<application-client-id>
```

#### Auth0

Auth0 strips non-namespaced custom claims from tokens, so groups and roles must be added via a post-login **Action** under full-URL claim keys:

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://example.com';
  api.accessToken.setCustomClaim(`${namespace}/groups`, event.user.groups || []);
  api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization?.roles || []);
};
```

```bash
MCP_AUTH_JWT_JWKS_URI=https://<tenant>.auth0.com/.well-known/jwks.json
MCP_AUTH_JWT_ISSUER=https://<tenant>.auth0.com/
MCP_AUTH_JWT_AUDIENCE=<api-identifier>
MCP_AUTH_GROUPS_CLAIM=https://example.com/groups
MCP_AUTH_ROLES_CLAIM=https://example.com/roles
```

Auth0 subjects carry a `|` (`auth0|...`, `google-oauth2|...`), which `ACCESS_CONTROL_DEFAULT_PRINCIPAL` cannot hold. To hand existing entries to such a subject, use the CLI route of [Switching an Existing Deployment to JWT](#switching-an-existing-deployment-to-jwt).

### Security Best Practices

1. **Pin issuer and audience**: leaving `MCP_AUTH_JWT_ISSUER`/`MCP_AUTH_JWT_AUDIENCE` unset skips those checks, so any token signed by the configured key is accepted
2. **Prefer JWKS mode** for real IdPs: key rotation at the IdP is picked up automatically
3. **Use HTTPS in production**: the JWT travels in the Authorization header and requires TLS
4. **Keep token lifetimes short**: the server checks `exp` on every request, but a token without `exp` never expires and `nbf` is not checked, so mint every token you issue yourself with a bounded `exp`
5. **Rotate long-running clients' tokens through a file**: `mcp-context-ingest --token-file PATH` reads the token from the file before every request, so a refresher can replace a short-lived token while a long sync runs; after a 401, the command retries the call once when the file holds a new token (see [The mcp-context-ingest CLI](knowledge-base.md#the-mcp-context-ingest-cli))

### Switching an Existing Deployment to JWT

Entries written over STDIO, `MCP_AUTH_PROVIDER=none` or `simple_token` are owned by `ACCESS_CONTROL_DEFAULT_PRINCIPAL` (default `local`). Under `jwt` every request runs as its token's subject, so no caller owns those entries: the private ones are unreachable and the public ones cannot be changed or deleted until they are handed to a real subject. Two routes exist:

1. **Set the default principal in the run that stamps the existing entries.** Set `ACCESS_CONTROL_DEFAULT_PRINCIPAL` to the operator's IdP subject in the environment of the run that writes the owner of the existing entries. For an integer-keyed v2 database, that run is the `mcp-context-server-migrate` conversion to the UUIDv7 schema: the CLI stamps every copied entry with the value from its own environment, and the first server start on the migrated database adds no column and changes no owner (see [Migrating to the UUIDv7 Schema](migration-v2-to-v3.md)). For a UUIDv7-keyed database whose `context_entries` table predates the ownership columns, that run is the first start of the upgraded server, which adds the columns and makes this subject the owner of every existing entry. Both runs reject any value outside 1-128 characters from `A-Z a-z 0-9 . _ @ : + -`, a limit that exists because the server interpolates the value into the access-control migration's DDL. Keycloak subjects (UUIDs) fit, and Auth0 subjects such as `auth0|...` and `google-oauth2|...` do not. Microsoft documents the Entra ID `sub` claim only as an opaque pairwise string, so compare the value your tokens carry against this character set before choosing this route.
2. **Reassign the owner with the migration CLI.** This route works for any subject, on a database that carries the ownership columns. Every database the UUIDv7 conversion produces carries them, so for an integer-keyed v2 database, run the conversion first and `--reassign-owner` on the migrated database. Stop the server, preview with `--dry-run`, then run:

   ```bash
   mcp-context-server-migrate --source-url <url> --reassign-owner local <subject>
   ```

   The CLI binds both values as statement parameters, so it has no character-set limit. It rewrites the owner of every entry owned by `local` in one statement, bumps `updated_at`, and keeps every grant on those entries (see [Sharing Entries](#sharing-entries)). With `--dry-run` it prints the number of matching entries and changes nothing. See [Reassigning Entry Ownership](migration-v2-to-v3.md#reassigning-entry-ownership).

### Anonymous Read Access

A `jwt` deployment can hand one shared token to readers who have no account at the identity provider, such as the readers of a public knowledge base. Set `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` to the subject that token carries, for example `anonymous`, and the server treats every request whose token carries that `sub` as a read-only principal:

- **Writes are refused.** `store_context`, `store_context_batch`, `update_context`, `update_context_batch`, `delete_context`, `delete_context_batch`, `share_context` and `unshare_context` answer `The anonymous principal is read-only: {tool} is not permitted` before they check any argument, look up any entry or generate anything. With `ENABLE_POSTGRESQL_RLS=true`, the database refuses the principal's inserts as well.
- **Reads work as for any principal.** The read tools, `list_context_grants` included, apply the usual rules. The principal owns no entry, so it reads the `public` entries, every returned entry reports `is_owner: false`, and `list_context_grants` answers it with not found or not authorized.
- **Groups and roles are discarded.** A token minted with a groups or roles claim by mistake reaches no group grant and never holds the publish role.
- **The token must expire.** The server refuses an anonymous-principal token without an `exp` claim with `The anonymous principal's token carries no expiry (exp); mint it with a bounded exp`.
- **Nobody shares with it.** `share_context` refuses a `user` grant to this principal, so through the tools an entry reaches anonymous readers only by being `public`.

The setting is compared verbatim with the token's `sub` claim, and it fails open: when it is unset, or differs from the subject you minted by one character, the token is an ordinary principal that stores, updates and deletes its own entries. The server logs `Read-only principal: <id>` at INFO level on startup, or `Read-only principal: none` when the setting is unset. The default `LOG_LEVEL=ERROR` hides that line, so to read it, start the server with `LOG_LEVEL=INFO` (or `DEBUG`) at least once after you set the principal. The [check before distribution](#checking-the-token-before-distributing-it) proves the match for each token. The value must differ from `ACCESS_CONTROL_DEFAULT_PRINCIPAL`, and an empty or whitespace-padded value is refused; either stops the server at startup with exit 78.

Clients send the token as a static header, `Authorization: Bearer <token>`. The server publishes no OAuth protected-resource metadata, so no client can discover a way to obtain a token by itself. `/health` answers without a token, as it does for every provider.

#### Minting the Token

The holder of the signing key mints the token: the private key behind `MCP_AUTH_JWT_PUBLIC_KEY` on a static-key deployment, or the private key behind `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` on a JWKS deployment (see [JWKS Deployments](#jwks-deployments)). For an `HS*` algorithm the key is the shared secret itself. The script below uses joserfc, a dependency of fastmcp that the server image and every server installation already carry; anywhere else, run it with `uv run --with joserfc mint.py`. Save it as `mint.py` beside `private.pem` (the PEM private key, or a file that holds only the `HS*` shared secret), set `ALGORITHM`, fill in the `iss` and `aud` values or delete those lines, and run it:

```python
import time
import uuid
from pathlib import Path

from joserfc import jwt
from joserfc.jwk import ECKey, OctKey, OKPKey, RSAKey

ALGORITHM = 'RS256'  # MCP_AUTH_JWT_ALGORITHM, or MCP_AUTH_ANONYMOUS_JWT_ALGORITHM on a JWKS deployment
KEY_CLASSES = {'RS': RSAKey, 'PS': RSAKey, 'ES': ECKey, 'Ed': OKPKey, 'HS': OctKey}
key = KEY_CLASSES[ALGORITHM[:2]].import_key(Path('private.pem').read_text().strip())
now = int(time.time())
claims = {
    'sub': 'anonymous',                # ACCESS_CONTROL_ANONYMOUS_PRINCIPAL
    'iss': '<MCP_AUTH_JWT_ISSUER>',    # only when the server sets MCP_AUTH_JWT_ISSUER
    'aud': '<MCP_AUTH_JWT_AUDIENCE>',  # only when the server sets MCP_AUTH_JWT_AUDIENCE
    'iat': now,
    'exp': now + 90 * 86400,           # always set exp, and mint the next token before it passes
    'jti': str(uuid.uuid4()),
}
print(jwt.encode({'alg': ALGORITHM}, claims, key, algorithms=[ALGORITHM]))
```

- `ALGORITHM` names the algorithm the server is configured to accept, and the script passes it to `jwt.encode`, because joserfc refuses algorithms it does not recommend, such as RS512, PS256 and HS512, unless the call names them. For an Ed25519 or Ed448 key, configure the server and the script with `Ed25519` or `Ed448`: joserfc warns about the generic `EdDSA` name, which RFC 9864 deprecates, although the server still accepts it.
- The script strips the key file, because the server strips `MCP_AUTH_JWT_PUBLIC_KEY` and `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY`, and an HS* secret read with a trailing newline would sign with a different key.
- The token carries no groups or roles claim. The server would discard both for this principal anyway.
- `jti` gives each token its own identity, so logs and audits can tell two tokens apart; the server does not check it.
- With an HS* algorithm the shared secret sits in the server configuration, so whoever can read that configuration can mint tokens: for any principal on a static-key deployment, and for the anonymous principal on a JWKS deployment. An asymmetric algorithm keeps the private key off the server.

#### Checking the Token Before Distributing It

Run this check with every token you mint, before you hand it out: a mistyped `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` leaves the token writable, and nothing else reports it. The script calls `store_context` with the token and passes only when the server refuses with the exact read-only text:

```python
import asyncio

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

URL = 'https://mcp.example.com/mcp'
TOKEN = '<the minted token>'
REFUSAL = 'The anonymous principal is read-only: store_context is not permitted'


async def main() -> None:
    transport = StreamableHttpTransport(URL, headers={'Authorization': f'Bearer {TOKEN}'})
    async with Client(transport) as client:
        try:
            result = await client.call_tool(
                'store_context',
                {'thread_id': 'anonymous-token-check', 'source': 'agent', 'text': 'Anonymous token check'},
            )
        except ToolError as error:
            if REFUSAL in str(error):
                print('The token is read-only: distribute it')
                return
            raise SystemExit(f'Do not distribute the token: {error}') from error
        await client.call_tool('delete_context', {'context_ids': [result.structured_content['context_id']]})
        raise SystemExit('Do not distribute the token: the server stored an entry for it, so it can write')


asyncio.run(main())
```

Any other outcome means the token must not be distributed. A stored entry means the setting is unset or differs from the token's `sub`; the script deletes that entry again. A 401 means the server rejects the token's signature, algorithm, issuer, audience or expiry. The expiry refusal means the token carries no `exp`.

#### JWKS Deployments

On a JWKS deployment the server trusts only tokens its identity provider signs, so you cannot mint the anonymous token with the provider's key. Add a second verifier whose key you hold:

1. Create a key pair, or an HS* shared secret, that only you hold. Set `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` to its PEM public key (or the secret) and `MCP_AUTH_ANONYMOUS_JWT_ALGORITHM` to its algorithm (default `RS256`).
2. Set `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` to a subject no user of the identity provider carries, such as `anonymous`. The second verifier accepts tokens for that subject only and rejects a correctly signed token for any other subject.
3. Pin `MCP_AUTH_JWT_AUDIENCE`, and `MCP_AUTH_JWT_ISSUER` where you can (see the JWT [Security Best Practices](#security-best-practices-1)). The second verifier checks the same issuer and audience as the JWKS verifier, so the `iss` and `aud` lines of the script carry the same values.
4. Mint the token with the script above, signing with the private key of step 1, and run the check.

The server then tries the anonymous verifier first and the JWKS verifier second, and logs `Authentication: JWTVerifier (JWKS endpoint, algorithm <alg>) composed with the anonymous static-key verifier (algorithm <alg>, subject <id>)` at startup. It refuses to start, with exit 78, when `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` is set without `MCP_AUTH_JWT_JWKS_URI` or without `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL`. On a static-key deployment, leave `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` unset and sign the anonymous token with the private key behind `MCP_AUTH_JWT_PUBLIC_KEY`.

#### Token Lifetime and Revocation

- **Bounded lifetime.** Always set `exp`. The example uses 90 days; keep the lifetime under a year, and mint and distribute the next token before the current one expires.
- **No revocation of one token.** The server keeps no list of tokens, so it cannot revoke a single anonymous token. Rotating the signing key (`MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` on a JWKS deployment, `MCP_AUTH_JWT_PUBLIC_KEY` on a static-key one) invalidates every token signed with the old key, and on a static-key deployment that includes the tokens of every other principal.
- **One shared identity.** Every holder of every anonymous token is the same principal. Everything `public` reaches all of them, and so does any grant to that principal that reaches the database without `share_context`, such as a row inserted with raw SQL.

#### Publishing to Anonymous Readers

Making an entry `public` exposes it to every anonymous token holder. On a deployment that hands out an anonymous token, set `ACCESS_CONTROL_PUBLISH_ROLE` so that only principals holding that role may publish. The role gates the `public` visibility only: it does not restrict `share_context`, and a group grant reaches every member of the group (see [Sharing Entries](#sharing-entries)).

## Access Model

Every request runs as one principal, and every entry belongs to the principal that stored it. The server enforces the model on every read, update and delete, on SQLite and PostgreSQL alike, and with `ENABLE_POSTGRESQL_RLS=true` PostgreSQL enforces it in the database as well (see [Enforcement and Its Limits](#enforcement-and-its-limits)).

### Principals

| Request                                                                                | Principal                                                                   | Groups and roles                                        |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------|
| STDIO transport                                                                        | `ACCESS_CONTROL_DEFAULT_PRINCIPAL` (default `local`)                        | None                                                    |
| HTTP with `MCP_AUTH_PROVIDER=none`                                                     | `ACCESS_CONTROL_DEFAULT_PRINCIPAL`                                          | None                                                    |
| HTTP with `MCP_AUTH_PROVIDER=simple_token`                                             | `ACCESS_CONTROL_DEFAULT_PRINCIPAL`                                          | None                                                    |
| HTTP with `MCP_AUTH_PROVIDER=jwt`                                                      | The token's `sub` claim (its client id when `sub` is absent)                | From `MCP_AUTH_GROUPS_CLAIM` and `MCP_AUTH_ROLES_CLAIM` |
| HTTP with `MCP_AUTH_PROVIDER=jwt`, `sub` equal to `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` | That `sub`, read-only (see [Anonymous Read Access](#anonymous-read-access)) | None: the server discards the token's groups and roles  |

Without `jwt`, every request shares the default principal, which owns every entry those requests store, so the server behaves as a single-user store. Under `jwt`, the client id is the token's `client_id` claim, or its `azp` claim when `client_id` is absent; a token that carries none of `sub`, `client_id` and `azp` runs as the principal `unknown`. The identity providers this guide covers put `sub` in every token.

### Ownership, Visibility and Grants

- **Owner**: the server stamps the writing principal as an entry's owner when the entry is first stored. The owner is never a tool parameter and no tool changes it; only the operator CLI `--reassign-owner` rewrites it.
- **Visibility**: `private` (readable by the owner and every grantee) or `public` (readable by everyone). An entry without an explicit `visibility` takes `ACCESS_CONTROL_DEFAULT_VISIBILITY` (default `private`).
- **Grants**: a grant gives one user or one group `read` or `write` permission on one entry. A grant is effective under either visibility, and a write grant also grants read. The entry's owner grants, removes and lists them with `share_context`, `unshare_context` and `list_context_grants`; a write grant does not allow sharing. `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups` writes a read grant for every group of the writing principal when an entry is first stored, so under the default `private` visibility the author's groups can read the entry.

| Operation                                                                                                                                                                                                                                                    | Allowed when the caller                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Read: `get_context_by_ids`, `search_context`, `semantic_search_context`, `fts_search_context`, `hybrid_search_context`, `grep_context`, `navigate_context`, `read_context_range`, `list_threads`, `get_statistics`, `plan_context_sync`, `list_ingest_state` | Owns the entry, the entry is `public`, or holds a read or write grant directly or through a group |
| Change text, metadata, tags or images: `update_context`, `update_context_batch`                                                                                                                                                                              | Owns the entry or holds a write grant directly or through a group                                 |
| Change visibility                                                                                                                                                                                                                                            | Owns the entry; publishing as `public` may also require `ACCESS_CONTROL_PUBLISH_ROLE`             |
| Delete: `delete_context`, `delete_context_batch`                                                                                                                                                                                                             | Owns the entry                                                                                    |
| Share, unshare and list grants: `share_context`, `unshare_context`, `list_context_grants`                                                                                                                                                                    | Owns the entry; a write grant does not allow sharing                                              |

The sync tools read under the same rule and narrow it further: `plan_context_sync` plans updates and deletes only for the caller's own managed entries and counts the other readable ones as `foreign_count`, and `list_ingest_state` lists only the caller's own (see [Access Scope](knowledge-base.md#access-scope)).

### Sharing Entries

Only an entry's owner shares it. `share_context` grants one user or one group `read` or `write` permission on one entry, `unshare_context` removes one grant, and `list_context_grants` lists every grant of the entry with the principal that granted it and when. A caller who reads or edits the entry through a grant, or reads it because it is `public`, gets an explicit refusal from all three tools (see [Not Found Versus Not Authorized](#not-found-versus-not-authorized)). Each call names one entry; no tool shares a whole thread. The [API Reference](api-reference.md#sharing-tools) lists the parameters and responses.

- **Idempotent shares.** Sharing a grant the entry already holds changes nothing and reports `created: false`. That includes a group read grant written by `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups`.
- **Separate permissions.** `read` and `write` are separate grants of the same principal, and a write grant also allows reading. `unshare_context` removes exactly the grant it names and reports `remaining_permissions`, the permissions the same principal type and ID still hold on the entry: removing `read` from a principal that also holds `write` leaves it able to read. The field covers only that principal type and ID; membership in a group that holds a grant can still give access.
- **Groups.** A group grant reaches every member of the group, as the token's groups claim lists them, including a group the identity provider gives every authenticated user. `ACCESS_CONTROL_PUBLISH_ROLE` gates only `visibility='public'`, never a group grant.
- **Refused grantees.** The owner cannot grant itself a `user` grant; a group grant to a group the owner belongs to is allowed. A `user` grant to the principal `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` names is refused: set the entry's visibility to `public` to publish it. `unshare_context` removes any grant the entry holds, including such grants written by other means.
- **Outside the update check.** Sharing and unsharing change neither the entry's `updated_at` nor the internal version that `update_context` compares before it writes, so a grant change never makes a concurrent update conflict. Tags-only and images-only updates also stay outside that check, but they do advance `updated_at`.
- **Lifetime.** Deleting an entry deletes its grants. The store tools write `author_groups` grants only when they insert a new entry, never when they deduplicate into an existing one, so unsharing such a grant is permanent. `mcp-context-server-migrate --reassign-owner` leaves an entry's grants in place: on PostgreSQL each grant also records the entry's owner, which the reassignment updates with the entry, and `granted_by` keeps naming the principal that created the grant, which can be the previous owner.

A knowledge-base corpus is shared one entry at a time; the [Knowledge Base Guide](knowledge-base.md) describes how a sync treats the grants of the entries it manages.

### Not Found Versus Not Authorized

- **An entry the caller may not read behaves exactly like one that does not exist.** It is never returned, matched, listed or counted; an ID prefix resolves over readable entries only; `update_context` reports `Context entry with ID {id} not found`, and `navigate_context` and `read_context_range` report `Context entry not found: {id}`.
- **An entry the caller may read but not modify is refused explicitly.** A caller who may only read the entry, through a read grant or because the entry is `public`, gets `Not authorized to modify context entry with ID {id}` from `update_context` (`update_context_batch`: `Not authorized to modify context entry {id}`) for every change, a visibility change included. A caller holding a write grant who is not the owner may change text, metadata, tags and images, and both update tools refuse that caller's visibility change with `Only the owner may change the visibility of context {id}`.
- **A delete that names IDs refuses the whole call.** When `delete_context` with `context_ids`, or `delete_context_batch` with `context_ids`, names an entry the caller may read but does not own, the call fails with `Not authorized to delete context entries: {ids}` and deletes nothing.
- **The sharing tools answer only the owner.** `share_context`, `unshare_context` and `list_context_grants` report an entry the caller may not read as `Context entry with ID {id} not found`, and refuse an entry the caller may read but does not own with `Not authorized to share context entry with ID {id}`, `Not authorized to unshare context entry with ID {id}` or `Not authorized to list the grants of context entry with ID {id}`. `share_context` also refuses `Cannot share context entry with ID {id} with its owner` and `Cannot share context entry with ID {id} with the anonymous principal; set its visibility to 'public' to publish it`.
- **The read-only principal is refused every write.** When `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` names the caller, `store_context`, `store_context_batch`, `update_context`, `update_context_batch`, `delete_context`, `delete_context_batch`, `share_context` and `unshare_context` fail with `The anonymous principal is read-only: {tool} is not permitted` before they check any argument or look up any entry. The read tools, `list_context_grants` included, serve that principal like any other caller.
- **Thread and criteria deletes cover only the caller's own entries.** `delete_context` with `thread_id` and `delete_context_batch` without `context_ids` delete the caller's own matching entries and leave every other entry in place without an error.
- **Deduplication merges only into the caller's own entries.** `store_context` and `store_context_batch` take as the deduplication candidate the latest entry of the same thread and source that the caller may read, and update it only when the caller owns it and the text is identical; otherwise they insert a new entry owned by the caller. Only opposite-source entries the caller may read mark a new conversational turn. A store with `deduplicate=false` looks for no candidate at all and always inserts a new entry owned by the caller, so it never touches another principal's entry either.

### Enforcement and Its Limits

The server applies the access rules in the application layer, on both backends: apart from the deployment-wide figures listed below, every query a tool runs against stored entries carries the access condition in SQL, ahead of any `LIMIT`, ranking or aggregation, and tags, images and per-node summaries are read only for entries such a query returned. Entries the caller cannot read therefore never take a slot on a page, in a ranked candidate set or in a top-N list.

On PostgreSQL, `ENABLE_POSTGRESQL_RLS=true` adds a database layer: row-level security policies, a guard trigger on ownership and visibility, and a read view apply the same rules to every request, with the caller's identity bound to each transaction, while the application condition stays in every query beside them. The policies name only a dedicated NOLOGIN policy role, so a database role outside it reads nothing, and the read view's role gate returns no row to any other role that obtains SELECT on it. Setup, the roles involved and the startup checks are described in [Row-level security (PostgreSQL)](database-backends.md#row-level-security-postgresql). Without row-level security, and always on SQLite, the database itself enforces nothing, and a client that connects to the database directly sees every row.

Row-level security guards against defects in how the server builds its queries: a missing or wrong access condition, a future query that forgets the caller's scope, or a statement that names the wrong relation. It does not guard against a compromised server process or injected SQL. The server asserts the caller's identity per transaction through `set_config`, as PostgREST and Supabase do with `request.jwt.claims`, and SQL running on the server's connection can set those values too.

Deferred generation writes derived data outside any request. A store or update made with `defer_generation=true` enqueues its generation job in the caller's own transaction, under the same rule as the write itself; with row-level security on, the insert policy of `generation_jobs` also admits a job only for an entry the caller may write. The generation worker then writes the entry's embeddings, summary and node summaries as the maintenance identity: on PostgreSQL with row-level security on, as the table owner on a connection of its own, which the policies do not filter, and otherwise with no identity bound. It writes only outputs generated from the entry's own stored text, only while the entry still holds that text, and changes neither the entry's text, metadata, tags, images, visibility and owner nor its grants. A job that a write grantee enqueued completes even when the grant is revoked before the worker runs it (see [Deferred Generation Jobs](database-backends.md#deferred-generation-jobs)).

The isolation has these limits:

- **SQLite full-text scores.** On SQLite, `fts_score` comes from FTS5 `bm25()`, which draws on statistics of the whole index (row count, average length, how many entries contain each term), entries the caller cannot read included. The scores of readable entries and their relative order can therefore shift as other principals' entries change, and the same score feeds the full-text leg of `hybrid_search_context`; which entries match, and how many results a page holds, depend only on what the caller can read. PostgreSQL's `ts_rank_cd` scores each entry on its own.
- **Deployment-wide statistics.** `get_statistics` reports `database_size_mb`, `embeddings_size_mb`, `embeddings_size_estimated`, `connection_metrics` (query counters and the last error text) and the configuration fields for the whole deployment, identically for every caller, an anonymous token holder included. Every entry, thread, image, tag, embedding, index, summary, node and generation job figure counts only the entries the caller may read.
- **Generation state.** Every entry-carrying read reports `generation_status` (`ready`, `pending` or `failed`) to every caller that may read the entry, so an anonymous token holder sees it on a `public` entry. The text of a job's failure, which can carry provider details such as a host name, stays in the job row and the server log, and no tool returns it to any caller. A pending or failed entry the caller may not read stays absent like any other: it is never returned, and its job is not counted in the caller's `generation` figures of `get_statistics`.
- **Public thread IDs.** `list_threads` lists every thread that holds an entry the caller may read, so a thread with one `public` entry shows its ID, and the figures of its readable entries, to every caller. Choose thread IDs that reveal nothing for threads that hold public entries.
- **Full-text migration progress.** While a full-text index rebuild runs, `fts_search_context` reports `estimated_remaining_seconds` computed from the total entry count.
- **Query plans.** With `explain_query=True` on PostgreSQL, the returned query plan carries row estimates from table-wide planner statistics.
- **Timing.** By timing many requests, a principal can roughly estimate how many entries it cannot read contain a chosen word or metadata value. The channel is coarse, noisy and content-dependent, and it exists with and without row-level security.
- **No leakproof barrier inside view reads.** With row-level security on, plain reads go through a view so that every search index stays usable, and PostgreSQL's leakproof protection therefore does not apply to them: the server's own conditions, such as a metadata filter, may be evaluated on entries the caller cannot read, as they are with row-level security off. No condition the server generates can raise an error on any stored value, and tests pin that.
- **Foreign-key checks.** PostgreSQL checks foreign keys without applying policies. Entry ids are server-generated UUIDv7 values, and an insert that names an entry the caller cannot read fails with the same error as one that names no entry, so the check reveals nothing.
- **Concurrent grant changes.** A statement evaluates grants as of its own snapshot, so a statement that runs while a grant is added or removed sees the grants as they were when it started. A tool call spans several statements, so an `unshare_context` that removes a caller's access while that caller's call runs shows up as a missing entry: a `navigate_context` call reports `Context entry not found: {id}` instead of an outline, and an `update_context` or `update_context_batch` call by a write grantee whose write grant goes away after the update checked it reports the entry as not found, even when the grantee can still read the entry through another grant or because it is `public`. With row-level security on, a tag, image or per-node summary write of such an update can instead fail with a generic error. Either way the update writes nothing.
- **One shared anonymous identity.** Every holder of an anonymous token is the same principal, reads the same entries and cannot be told apart by the server, and one token cannot be revoked without rotating the signing key (see [Token Lifetime and Revocation](#token-lifetime-and-revocation)).
- **The read-only setting fails open.** `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` makes a principal read-only only when it equals the token's `sub` exactly; unset or mistyped, the anonymous token can write. The startup line `Read-only principal: <id>` (logged at INFO, so the default `LOG_LEVEL=ERROR` hides it) and the [check before distribution](#checking-the-token-before-distributing-it) expose a mismatch.
- **The publish role gates visibility only.** `ACCESS_CONTROL_PUBLISH_ROLE` decides who may make an entry `public`. It does not limit who may share: an owner can grant any user or group, and a group grant reaches every member of the group, including a group the identity provider gives every authenticated user.
- **Refused calls are logged.** FastMCP logs every refused tool call at ERROR level as `Error calling tool '<name>'`, without a traceback, so a public endpoint whose anonymous clients attempt writes logs one ERROR line per refused call.

For deployments with more than one principal, use PostgreSQL with row-level security enabled.

## MCP Client Configuration

### Claude Desktop

Claude Desktop configuration varies by authentication method:

**STDIO (No Auth):**

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-context-server"]
    }
  }
}
```

**HTTP (No Auth):**

```json
{
  "mcpServers": {
    "context-server": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**HTTP with Bearer Token:**

Check Claude Desktop documentation for the latest authentication header support. As of this writing, Claude Desktop's HTTP transport may have limited support for custom authentication headers.

### Claude Code CLI

```bash
# Add STDIO server (no auth)
claude mcp add context-server -- uvx mcp-context-server

# Add HTTP server (no auth)
claude mcp add --transport http context-server http://localhost:8000/mcp

# Add HTTP server with Bearer token authentication
claude mcp add --transport http context-server http://localhost:8000/mcp --header "Authorization: Bearer your-secret-token-here"
```

### Custom MCP Clients

For custom clients implementing MCP protocol:

**Bearer Token:**
```python
# Include in all requests
headers = {"Authorization": f"Bearer {token}"}
```

## Environment Variables Reference

### Bearer Token Authentication

| Variable             | Required | Default      | Description                                                   |
|----------------------|----------|--------------|---------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`  | Yes      | -            | `simple_token`                                                |
| `MCP_AUTH_TOKEN`     | Yes      | -            | Bearer token for validation                                   |
| `MCP_AUTH_CLIENT_ID` | No       | `mcp-client` | Client ID that identifies the client to FastMCP; not an owner |

### JWT Authentication

| Variable                            | Required            | Default  | Description                                                                                      |
|-------------------------------------|---------------------|----------|--------------------------------------------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`                 | Yes                 | -        | `jwt`                                                                                            |
| `MCP_AUTH_JWT_JWKS_URI`             | One of the key pair | -        | JWKS endpoint URI                                                                                |
| `MCP_AUTH_JWT_PUBLIC_KEY`           | One of the key pair | -        | PEM public key or HS* shared secret                                                              |
| `MCP_AUTH_JWT_ISSUER`               | No                  | -        | Expected `iss` claim value                                                                       |
| `MCP_AUTH_JWT_AUDIENCE`             | No                  | -        | Expected `aud` claim value                                                                       |
| `MCP_AUTH_JWT_ALGORITHM`            | No                  | `RS256`  | Accepted signing algorithm                                                                       |
| `MCP_AUTH_GROUPS_CLAIM`             | No                  | `groups` | Claim key for group memberships                                                                  |
| `MCP_AUTH_ROLES_CLAIM`              | No                  | `roles`  | Claim key for roles                                                                              |
| `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY` | No                  | -        | PEM public key or HS* shared secret of the anonymous verifier; JWKS deployments only             |
| `MCP_AUTH_ANONYMOUS_JWT_ALGORITHM`  | No                  | `RS256`  | Signing algorithm the anonymous verifier accepts                                                 |

### Access Control

| Variable                             | Required | Default | Description                                                                                          |
|--------------------------------------|----------|---------|------------------------------------------------------------------------------------------------------|
| `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` | No       | -       | The `jwt` subject every write tool refuses; compared verbatim with `sub`, so a mismatch fails open   |
| `ACCESS_CONTROL_PUBLISH_ROLE`        | No       | -       | Role a principal must hold to make an entry `public`; recommended with an anonymous token            |

The other access-control settings (`ACCESS_CONTROL_DEFAULT_PRINCIPAL`, `ACCESS_CONTROL_DEFAULT_VISIBILITY`, `ACCESS_CONTROL_DEFAULT_GROUP_GRANTS` and the row-level security settings) are described in the [Environment Variables Reference](environment-variables.md).

## Troubleshooting

### Issue 1: "MCP_AUTH_TOKEN is required" Error

**Symptom:** Server fails to start with token error

**Cause:** `MCP_AUTH_PROVIDER` is set to `simple_token` but `MCP_AUTH_TOKEN` is not set

**Solution:**
```bash
# Set the token
export MCP_AUTH_TOKEN=your-secret-token

# Or disable auth
export MCP_AUTH_PROVIDER=none
```

### Issue 2: Bearer Token Rejected

**Symptom:** HTTP 401 Unauthorized despite correct token

**Causes:**
- Token mismatch (check for trailing whitespace/newlines)
- Missing "Bearer " prefix in header
- Token not properly URL-encoded if special characters

**Solutions:**
```bash
# Verify exact token match
echo -n "$MCP_AUTH_TOKEN" | xxd

# Test with curl
curl -v -H "Authorization: Bearer $MCP_AUTH_TOKEN" http://localhost:8000/mcp
```

### Issue 3: JWT Rejected Despite Valid IdP Login

**Symptom:** HTTP 401 Unauthorized with a freshly issued IdP token

**Causes:**
- Issuer mismatch: the token `iss` differs from `MCP_AUTH_JWT_ISSUER` (trailing slash matters)
- Audience mismatch: the token `aud` does not contain `MCP_AUTH_JWT_AUDIENCE`
- Algorithm mismatch: the IdP signs with a different algorithm than `MCP_AUTH_JWT_ALGORITHM`
- Expired token: the `exp` claim is in the past

**Solutions:** decode the token payload (for example at jwt.io or with `python -c "import base64,json,sys; p=sys.argv[1].split('.')[1]; print(json.dumps(json.loads(base64.urlsafe_b64decode(p+'='*(-len(p)%4))), indent=2))" "$TOKEN"`) and compare `iss`, `aud`, `exp`, and the header `alg` against the configured values. Server-side rejection reasons are logged at WARNING level.

### Issue 4: Empty Groups With Entra ID

**Symptom:** Authentication succeeds but the caller has no group memberships, and the log warns about group overage

**Cause:** the user belongs to more groups than Entra ID embeds in a token, so the `groups` claim was replaced by overage markers; the server fails closed to an empty group list and never calls Microsoft Graph

**Solution:** in the app registration, emit only groups assigned to the application (or otherwise filter the group claims) so the token stays under the limit

### Common Error Messages

| Error                                                                           | Cause                                                                                | Solution                                                                                |
|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `MCP_AUTH_TOKEN cannot be empty`                                                | Token set to empty string                                                            | Provide valid token or remove auth                                                      |
| `Token validation failed`                                                       | Token mismatch                                                                       | Verify token matches exactly                                                            |
| `requires exactly one of ...`                                                   | Neither JWT key source configured                                                    | Set `MCP_AUTH_JWT_PUBLIC_KEY` or `MCP_AUTH_JWT_JWKS_URI`                                |
| `... are mutually exclusive`                                                    | Both JWT key sources configured                                                      | Unset one of the two key sources                                                        |
| `Unsupported algorithm`                                                         | Invalid `MCP_AUTH_JWT_ALGORITHM` or `MCP_AUTH_ANONYMOUS_JWT_ALGORITHM`               | Use a supported HS*/RS*/ES*/PS*/Ed* value                                               |
| `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY applies to MCP_AUTH_JWT_JWKS_URI ...`        | The anonymous verifier key is set on a static-key deployment                         | Unset it and sign the anonymous token with the private key of `MCP_AUTH_JWT_PUBLIC_KEY` |
| `MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY requires ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` | The anonymous verifier key is set without the anonymous principal                    | Set `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` to the subject the token carries               |
| `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL must ...`                                   | The value is empty, whitespace-padded or equal to `ACCESS_CONTROL_DEFAULT_PRINCIPAL` | Set the exact `sub` of the anonymous token, distinct from the default principal         |
| `The anonymous principal is read-only: {tool} is not permitted`                 | A write by the principal `ACCESS_CONTROL_ANONYMOUS_PRINCIPAL` names                  | Expected for the anonymous token; write with another principal's token                  |
| `The anonymous principal's token carries no expiry (exp) ...`                   | The anonymous token was minted without `exp`                                         | Mint a new token with a bounded `exp` (see [Minting the Token](#minting-the-token))     |

## Security Recommendations

### For Bearer Token

1. **Generate strong tokens**: Use `secrets.token_urlsafe(32)` minimum
2. **Use HTTPS**: Required for production to protect token in transit
3. **Rotate periodically**: Change tokens on regular schedule
4. **Limit scope**: Use separate tokens for different services
5. **Monitor usage**: Log authentication events for audit

### General Best Practices

1. **HTTPS everywhere**: Use TLS for all HTTP transport deployments
2. **Principle of least privilege**: Grant minimum necessary access
3. **Audit logging**: Enable logging for authentication events
4. **Regular rotation**: Rotate secrets and tokens periodically
5. **Secure storage**: Use secrets managers for credentials

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata filtering with operators
- **Knowledge Base**: [Knowledge Base Guide](knowledge-base.md) - document corpora, their visibility and sharing them
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - HTTP transport configuration
- **Main Documentation**: [README.md](../README.md) - overview and quick start
- **FastMCP Authentication**: [FastMCP Auth](https://gofastmcp.com/servers/auth) - FastMCP auth documentation

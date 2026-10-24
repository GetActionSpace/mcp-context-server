# Security Policy

## Supported Versions

| Version |          Supported |
|---------|-------------------:|
| 3.x     | :white_check_mark: |
| < 3.0   |                :x: |

> **Policy:** Only the latest major version receives security updates. When a new major version is released, prior major versions are immediately unsupported.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **DO NOT** open a public issue
2. Send details via [GitHub Security Advisories](https://github.com/GetActionSpace/mcp-context-server/security/advisories/new)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Best Practices

### For Users

1. **Use Authentication for HTTP Transport**
   - Enable bearer token authentication when exposing the server over HTTP
   - Set `MCP_AUTH_PROVIDER=simple_token` and configure `MCP_AUTH_TOKEN`
   - Authentication is not required for stdio transport (local only)

2. **Use TLS in Production**
   - The server does not terminate TLS natively
   - Deploy behind a TLS-terminating reverse proxy (nginx, Caddy, Traefik) for any network-accessible deployment

3. **Protect Database Credentials**
   - Use environment variables or secrets management for PostgreSQL and Supabase connection strings
   - Never hardcode database credentials in configuration files

4. **Container Security**
   - The official Docker image runs as non-root (UID 10001)
   - Use read-only filesystem mounts where possible
   - Follow the principle of least privilege for container permissions

### For Contributors

1. **Never Commit Secrets**
   - API keys
   - Passwords
   - Database connection strings
   - Private certificates

2. **Validate Input**
   - Prevent SQL injection via parameterized queries
   - Validate and sanitize all tool parameters
   - Use centralized configuration management for credentials

3. **Use Secure Defaults**
   - Restrictive default configuration
   - Opt-in for network exposure (stdio is default, HTTP requires explicit configuration)
   - Explicit permissions over wildcards

## Security Updates

Security patches will be released as soon as possible after discovery. Watch this repository for updates.

## Compliance

This project aims to follow security best practices including:
- OWASP guidelines where applicable
- Principle of least privilege
- Defense in depth
- Secure by default

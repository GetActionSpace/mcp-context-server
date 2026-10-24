# Contributing to MCP Context Server

## Contributor Terms

MCP Context Server is licensed under the [Elastic License 2.0](LICENSE) and is dual-licensed commercially by the project owner (see [Commercial Licensing](docs/commercial-licensing.md)). To keep that model legally sound, every external contribution must grant the project owner the right to license the contributed code under any terms, including commercial ones.

By submitting a pull request, you agree that: (1) the contribution is your original work (or you have the right to submit it); (2) you grant Aleksandr Filippov a perpetual, worldwide, non-exclusive, irrevocable, royalty-free license to use, reproduce, modify, distribute, sublicense, and relicense your contribution under any license terms, including the Elastic License 2.0 and separate commercial licenses; and (3) you retain copyright in your contribution. A Developer Certificate of Origin sign-off alone is not sufficient, because it does not convey relicensing rights.

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Run pre-commit hooks** before committing
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## Development Workflow

### Local Development Setup

```bash
# 1. Clone and setup
git clone https://github.com/GetActionSpace/mcp-context-server.git
cd mcp-context-server
uv sync --all-extras --all-groups

# 2. Run tests
uv run pytest                         # Unit tests
uv run pytest -m integration          # Integration tests

# 3. Test server locally
uv run python -m app.server           # Should start without errors
# Press Ctrl+C to stop

# 4. Test published version (optional)
uvx --python 3.12 mcp-context-server  # Run from PyPI without cloning
```

> **Important**: Development requires ALL dependencies installed. Always use
> `uv sync --all-extras --all-groups` instead of bare `uv sync`. This ensures
> all optional provider packages, type stubs, and development tools are available.
> Both local development and CI use this command. Running bare `uv sync` may cause
> type-checker errors due to missing optional dependencies.

### Making Changes

After code changes:
1. Test your changes: `uv run pytest`
2. Run code quality checks: `uv run pre-commit run --all-files`

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/tools/context/test_store.py

# Run metadata filtering tests
uv run pytest tests/core/test_query_builder.py tests/core/test_query_builder_type_guards.py tests/core/test_query_builder_numeric.py tests/core/test_query_builder_array_contains.py tests/core/test_query_builder_limits.py tests/core/test_metadata_types.py -v
uv run pytest tests/tools/search/test_browse_metadata_filters.py tests/tools/search/test_browse_array_contains.py tests/tools/test_nested_metadata_storage.py -v
uv run pytest tests/tools/test_metadata_error_handling.py -v

# Run semantic search tests
uv run pytest tests/repositories/embedding_repository tests/tools/search/test_semantic.py -v

# Run date filtering tests
uv run pytest tests/startup/test_validation.py tests/tools/search/test_browse_date_filtering.py tests/tools/search/test_semantic_date_filtering.py tests/repositories/test_base.py tests/repositories/context_repository/test_search_sqlite_datetime.py -v

# Run integration tests only
uv run pytest -m integration

# Skip slow tests for quick feedback
uv run pytest -m "not integration"
```

### Code Quality

```bash
# Run pre-commit hooks on all files, including Ruff, mypy, and pyright
uv run pre-commit run --all-files
```

### Continuous Integration Gates

Every pull request runs `Test` and `Test Summary` (`.github/workflows/test.yml`) plus `Run Pre-Commit Checks`, `MCP Checks` and `Security Scan` (`.github/workflows/lint.yml`). The only exemption is Release Please's own version-bump PR, recognized by BOTH the `release-please--branches--` head-branch prefix AND the PR author being the repository owner; the branch prefix alone is contributor-controlled and never sufficient on its own.

**Repository setting maintainers must keep in place:** these five jobs belong on the `main` branch ruleset as REQUIRED status checks. Without that, a job that is skipped rather than run shows up as neutral in the Checks tab, so a pull request whose CI never executed presents as having nothing failing. The workflow conditions decide *when* a job runs; only the ruleset makes a green result a merge requirement. This is a repository-settings change and cannot be made from the working tree.

## Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for all commit messages. This enables automated versioning and changelog generation.

### Commit Types

#### `feat`: New Features
Introduces new functionality or capabilities to the MCP Context Server.

Examples:
- `feat: add batch context retrieval endpoint`
- `feat: implement context search by date range`
- `feat: support WebP image format in multimodal storage`
- `feat: add metadata filtering with 15 operators`

#### `fix`: Bug Fixes
Fixes issues or bugs in the existing codebase.

Examples:
- `fix: resolve database lock on concurrent writes`
- `fix: handle invalid base64 image data gracefully`
- `fix: prevent memory leak in connection pool`

#### `chore`: Maintenance Tasks
Updates dependencies, refactors code, or performs housekeeping tasks.

Examples:
- `chore: update FastMCP to version 2.13`
- `chore: reorganize repository module structure`
- `chore: clean up unused test fixtures`

#### `docs`: Documentation
Improves or updates documentation, including README, API docs, or code comments.

Examples:
- `docs: add examples for multimodal context storage`
- `docs: update MCP client configuration guide`
- `docs: clarify thread-based context scoping in architecture`

#### `ci`: CI/CD Changes
Modifies continuous integration, deployment pipelines, or automation workflows.

Examples:
- `ci: add automated PyPI release workflow`
- `ci: configure pre-commit hooks for type checking`
- `ci: enable coverage reporting in GitHub Actions`

#### `test`: Testing
Adds or modifies tests, including unit, integration, or end-to-end tests.

Examples:
- `test: add integration tests for thread isolation`
- `test: implement concurrent write test scenarios`
- `test: validate multimodal context deduplication`
- `test: add comprehensive metadata filtering tests`

### Version Bump Rules

The commit type determines how the version number is incremented:

- **`feat:`** → Minor version bump (`0.x.0`)
- **`fix:`** → Patch version bump (`0.0.x`)
- **`feat!`**, **`fix!`**, or **`BREAKING CHANGE`** → Major version bump (`x.0.0`)

### Commit Message Format

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Examples:**

```text
feat: add support for compressed image storage

Implements automatic image compression for attachments larger than 1MB, reducing database size by up to 70% while maintaining visual quality.

Closes #42
```

```text
fix: resolve race condition in repository initialization

The database connection manager now uses a lock to prevent concurrent initialization attempts during high load.
```

## Release Process

Releases are automated using [Release Please](https://github.com/googleapis/release-please).

### How It Works

1. **Conventional commits** on `main` branch are tracked automatically
2. Release Please creates/updates a release PR with changelog
3. Merging the release PR:
   - Creates a GitHub release with semantic version tag
   - Triggers `publish.yml` workflow via `release:published` event

### Publish Workflow

On release, three jobs run from `.github/workflows/publish.yml`:

| Job                       | Depends On        | Output                                                                                                  |
|---------------------------|-------------------|---------------------------------------------------------------------------------------------------------|
| `publish-to-pypi`         | `build`           | Package on [PyPI](https://pypi.org/project/mcp-context-server/)                                         |
| `publish-docker-image`    | `build`           | Image on [GHCR](https://github.com/GetActionSpace/mcp-context-server/pkgs/container/mcp-context-server) |
| `publish-to-mcp-registry` | `publish-to-pypi` | Entry in [MCP Registry](https://registry.modelcontextprotocol.io/)                                      |

PyPI and Docker publishing run in parallel. MCP Registry waits for PyPI.

### Docker Image Details

**Registry:** `ghcr.io/getactionspace/mcp-context-server`

**Platforms:** `linux/amd64`, `linux/arm64`

**Tags** (for release `v0.14.0`):
- `0.14.0` - Full version
- `0.14` - Minor version
- `0` - Major version
- `sha-abc1234` - Git commit SHA
- `latest` - Most recent release

**Supply Chain Security:**
- SLSA provenance attestation
- Software Bill of Materials (SBOM)
- Cryptographically signed build provenance via `actions/attest-build-provenance`

## Release Troubleshooting

### Recovering from Partial Release (PyPI Success, MCP Registry Failure)

If a new version was successfully published to PyPI but failed to publish to MCP Registry, follow this recovery procedure:

1. **Fix the root cause**: Identify and fix whatever prevented MCP Registry publication (e.g., `server.json` schema validation errors, network issues).

2. **Commit and push/merge fixes without triggering Release Please**: Use commit types that do NOT trigger a new release. Avoid `feat` and `fix` prefixes, as these trigger Release Please to propose a new version with an incorrect changelog.

   Allowed commit types for fixes:
   - `chore:` - Maintenance tasks
   - `ci:` - CI/CD changes
   - `docs:` - Documentation updates
   - `test:` - Test modifications

   Example:
   ```bash
   git add server.json
   git commit -m "chore: fix server.json schema for MCP Registry"
   git push origin main
   ```

3. **Delete the remote tag**: This makes the already-published GitHub release become a draft.
   ```bash
   git push origin :refs/tags/v0.10.0
   ```

4. **Create a new local tag**: Point the tag to the latest commit (with your fixes).
   ```bash
   git tag -fa v0.10.0 -m "v0.10.0"
   ```

5. **Push the updated tag**:
   ```bash
   git push origin v0.10.0
   ```

6. **Re-publish the GitHub release**: In the GitHub UI, navigate to Releases, find the draft release, and click "Publish release". This triggers the publish workflow:
   - PyPI publication is **skipped** (version already exists due to `skip-existing: true`)
   - Docker image is **rebuilt and pushed** (tags are overwritten)
   - MCP Registry publication proceeds with the fixed files

**Important**: Replace `v0.10.0` with your actual version tag in all commands.

## Need Help?

- Check existing [GitHub Issues](https://github.com/GetActionSpace/mcp-context-server/issues)
- Read the [README.md](README.md) for usage documentation
- Create a new issue with detailed information if needed

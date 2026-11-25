"""Server runtime settings: log level, disabled tools, transport, and the server instructions."""

from typing import Literal

from pydantic import Field

from app.settings.base import CommonSettings


class LoggingSettings(CommonSettings):
    """Application logging configuration."""

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='ERROR',
        alias='LOG_LEVEL',
        description='Application log level',
    )


class ToolManagementSettings(CommonSettings):
    """MCP tool availability configuration."""

    disabled_raw: str = Field(
        default='',
        alias='DISABLED_TOOLS',
        description='Comma-separated list of tools to disable (e.g., delete_context,update_context)',
    )

    @property
    def disabled(self) -> set[str]:
        """Parse comma-separated string into lowercase set of disabled tool names."""
        if not self.disabled_raw or not self.disabled_raw.strip():
            return set()
        return {t.lower().strip() for t in self.disabled_raw.split(',') if t.strip()}


class TransportSettings(CommonSettings):
    """HTTP transport settings for Docker/remote deployments."""

    transport: Literal['stdio', 'http', 'streamable-http', 'sse'] = Field(
        default='stdio',
        alias='MCP_TRANSPORT',
        description='Transport mode: stdio for local, http for Docker/remote',
    )
    host: str = Field(
        default='0.0.0.0',
        alias='FASTMCP_HOST',
        description='HTTP bind address (use 0.0.0.0 for Docker)',
    )
    port: int = Field(
        default=8000,
        alias='FASTMCP_PORT',
        ge=1,
        le=65535,
        description='HTTP port number',
    )
    stateless_http: bool = Field(
        default=True,
        alias='FASTMCP_STATELESS_HTTP',
        description='Enable stateless HTTP mode for horizontal scaling. '
                    'Enabled by default as the server has no stateful MCP features. '
                    'Set to false only if you need server-side MCP session tracking.',
    )
    keep_alive_timeout_s: int = Field(
        default=75,
        alias='HTTP_KEEP_ALIVE_TIMEOUT_S',
        ge=1,
        description='Seconds the HTTP server keeps an idle kept-alive connection open. '
                    'Keep it above the idle timeout of any proxy or load balancer in front of the server '
                    'and above the idle expiry of the HTTP clients that call it.',
    )


class InstructionsSettings(CommonSettings):
    """Server instructions sent to MCP clients during initialization."""

    server_instructions: str | None = Field(
        default=None,
        alias='MCP_SERVER_INSTRUCTIONS',
        description='Custom server instructions text. Overrides the built-in default instructions. '
                    'Set to empty string to disable instructions entirely.',
    )

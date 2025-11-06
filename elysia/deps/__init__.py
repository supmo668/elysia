"""
Dependency injection module for Elysia.
Provides health checks and other injectable dependencies.
"""

from elysia.deps.mcp_health import MCPHealthChecker

__all__ = ["MCPHealthChecker"]

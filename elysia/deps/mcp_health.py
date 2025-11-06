"""
MCP Health Checker - Dependency injection for MCP server health validation.

Provides health checking for MCP servers before initialization to prevent
failures when servers are not running.
"""
import asyncio
from logging import getLogger
from typing import Optional
from urllib.parse import urlparse, urljoin

logger = getLogger(__name__)

# Try to import aiohttp, but gracefully handle if it's not available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - MCP health checks will be skipped. Install with: pip install aiohttp")


class MCPHealthChecker:
    """
    Health checker for MCP servers using dependency injection pattern.
    
    Usage:
        checker = MCPHealthChecker()
        is_healthy = await checker.check_health(url="http://localhost:8000/sse")
    """
    
    def __init__(
        self,
        timeout: float = 3.0,
        health_endpoint: str = "/health",
        fallback_endpoints: Optional[list[str]] = None
    ):
        """
        Initialize MCP health checker.
        
        Args:
            timeout: Timeout in seconds for health check requests
            health_endpoint: Primary health check endpoint path
            fallback_endpoints: Alternative endpoints to try if primary fails
        """
        self.timeout = timeout
        self.health_endpoint = health_endpoint
        self.fallback_endpoints = fallback_endpoints or ["/mcp/health", "/api/health", "/.well-known/health"]
    
    async def check_health(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None
    ) -> bool:
        """
        Check if MCP server is healthy and responding.
        
        Args:
            url: Base URL of the MCP server (e.g., "http://localhost:8000/sse")
            headers: Optional HTTP headers for the request
            
        Returns:
            True if server is healthy, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - skipping health check, assuming server is healthy")
            return True
        
        # Parse URL to get base
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Try primary health endpoint
        if await self._check_endpoint(base_url, self.health_endpoint, headers):
            logger.debug(f"MCP server at {base_url} is healthy (primary endpoint)")
            return True
        
        # Try fallback endpoints
        for fallback in self.fallback_endpoints:
            if await self._check_endpoint(base_url, fallback, headers):
                logger.debug(f"MCP server at {base_url} is healthy (fallback: {fallback})")
                return True
        
        # Try the original URL path itself as a connectivity check
        if await self._check_endpoint(base_url, parsed.path, headers):
            logger.debug(f"MCP server at {base_url} is reachable (original path: {parsed.path})")
            return True
        
        logger.warning(f"MCP server at {base_url} is not responding to health checks")
        return False
    
    async def _check_endpoint(
        self,
        base_url: str,
        path: str,
        headers: Optional[dict[str, str]] = None
    ) -> bool:
        """
        Check a specific endpoint.
        
        Args:
            base_url: Base URL of the server
            path: Path to check
            headers: Optional HTTP headers
            
        Returns:
            True if endpoint responds successfully, False otherwise
        """
        try:
            endpoint = urljoin(base_url, path)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    # Accept 2xx, 3xx, and even 404 (server is up but endpoint doesn't exist)
                    # Reject 5xx (server errors) and connection failures
                    if response.status < 500:
                        return True
                    
        except aiohttp.ClientConnectorError:
            logger.debug(f"Connection failed to {base_url}{path}")
        except asyncio.TimeoutError:
            logger.debug(f"Timeout checking {base_url}{path}")
        except Exception as e:
            logger.debug(f"Error checking {base_url}{path}: {e}")
        
        return False
    
    async def check_multiple(
        self,
        servers: list[dict]
    ) -> dict[str, bool]:
        """
        Check health of multiple MCP servers concurrently.
        
        Args:
            servers: List of server configs with 'name', 'url', and optional 'headers'
            
        Returns:
            Dictionary mapping server names to health status
        """
        tasks = []
        server_names = []
        
        for server in servers:
            name = server.get("name", "unknown")
            url = server.get("url")
            headers = server.get("headers")
            
            if url:
                server_names.append(name)
                tasks.append(self.check_health(url, headers))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for name, result in zip(server_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error checking health of {name}: {result}")
                health_status[name] = False
            else:
                health_status[name] = result
        
        return health_status


# Global singleton instance
_health_checker: Optional[MCPHealthChecker] = None


def get_health_checker() -> MCPHealthChecker:
    """
    Get or create the global MCP health checker instance.
    
    Returns:
        Global MCPHealthChecker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = MCPHealthChecker()
    return _health_checker

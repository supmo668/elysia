"""
Test script for MCP health checker.
Validates that the health check system works correctly.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from elysia.deps.mcp_health import MCPHealthChecker, get_health_checker


async def test_health_checker():
    """Test the MCP health checker."""
    print("=" * 80)
    print("MCP Health Checker Test")
    print("=" * 80)
    
    checker = get_health_checker()
    
    # Test 1: Check a server that should be down
    print("\n[Test 1] Checking non-existent server...")
    result = await checker.check_health("http://localhost:9999/sse")
    print(f"Result: {'✓ Healthy' if result else '✗ Unhealthy'} (expected: ✗ Unhealthy)")
    
    # Test 2: Check the Graphiti server (if running)
    print("\n[Test 2] Checking Graphiti server at localhost:8000...")
    result = await checker.check_health("http://localhost:8000/sse")
    print(f"Result: {'✓ Healthy' if result else '✗ Unhealthy'}")
    
    # Test 3: Check multiple servers
    print("\n[Test 3] Checking multiple servers concurrently...")
    servers = [
        {
            "name": "graphiti-memory",
            "url": "http://localhost:8000/sse",
        },
        {
            "name": "non-existent",
            "url": "http://localhost:9999/sse",
        }
    ]
    
    results = await checker.check_multiple(servers)
    print("Results:")
    for name, healthy in results.items():
        status = "✓ Healthy" if healthy else "✗ Unhealthy"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 80)
    print("Health Checker Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_health_checker())

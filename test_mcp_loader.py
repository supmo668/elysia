"""
Test script to verify MCP loader with health checks.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from elysia.tools.mcp.mcp_loader import load_mcp_servers_from_config


def test_mcp_loader():
    """Test the MCP loader with health checks."""
    print("=" * 80)
    print("MCP Loader with Health Checks Test")
    print("=" * 80)
    print()
    
    print("Loading MCP servers from mcp.json...")
    print()
    
    tool_classes = load_mcp_servers_from_config()
    
    print()
    print("=" * 80)
    print(f"Loaded {len(tool_classes)} tool classes")
    print("=" * 80)
    
    if tool_classes:
        print("\nTool classes:")
        for tool_class in tool_classes:
            print(f"  - {tool_class.__name__}")
    else:
        print("\nNo tool classes loaded (servers may be down or unhealthy)")
    
    print()


if __name__ == "__main__":
    test_mcp_loader()

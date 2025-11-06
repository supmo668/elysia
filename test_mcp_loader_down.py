"""
Test script to simulate MCP server down scenario.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_server_down():
    """Test MCP loader when server is down."""
    print("=" * 80)
    print("MCP Loader Test - Server Down Scenario")
    print("=" * 80)
    print()
    
    # Create a temporary mcp.json with a server that doesn't exist
    temp_config = {
        "servers": [
            {
                "name": "non-existent-server",
                "description": "A server that doesn't exist",
                "type": "sse",
                "url": "http://localhost:9999/sse",
                "enabled": True
            }
        ]
    }
    
    config_path = Path(__file__).parent / "mcp.json.bak"
    original_path = Path(__file__).parent / "mcp.json"
    
    # Backup original
    if original_path.exists():
        with open(original_path, 'r') as f:
            original_config = f.read()
    else:
        original_config = None
    
    try:
        # Write temp config
        with open(original_path, 'w') as f:
            json.dump(temp_config, f, indent=2)
        
        print("Created temporary mcp.json with non-existent server...")
        print()
        
        # Import and test
        from elysia.tools.mcp.mcp_loader import load_mcp_servers_from_config
        
        tool_classes = load_mcp_servers_from_config()
        
        print()
        print("=" * 80)
        print(f"Result: {len(tool_classes)} tool classes loaded")
        print("=" * 80)
        
        if len(tool_classes) == 0:
            print("\n✓ SUCCESS: No tools loaded when server is down (expected behavior)")
        else:
            print("\n✗ FAILURE: Tools were loaded even though server is down")
        
    finally:
        # Restore original
        if original_config:
            with open(original_path, 'w') as f:
                f.write(original_config)
            print("\n✓ Restored original mcp.json")
    
    print()


if __name__ == "__main__":
    test_server_down()

#!/usr/bin/env python3
"""Test script to verify MCP tool loading during tree initialization."""

import logging
import sys

# Set up logging to see INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Import after logging setup
from elysia.tree.tree import Tree
from elysia.tree.objects import Settings

print("=" * 80)
print("Starting Tree Initialization Test")
print("=" * 80)

# Create settings
settings = Settings()
settings.logger = logging.getLogger("elysia")

print("\n--- Creating Tree with 'one_branch' mode ---\n")

# Initialize tree with one_branch mode (default mode)
tree = Tree(settings=settings, branch_initialisation="one_branch")

print("\n--- Tree Initialized ---\n")

# Check if MCP tools were loaded
print("Checking for MCP tools in tree...")
mcp_tools_found = []
for branch_id in tree.branch_ids:
    branch = tree.get_branch(branch_id)
    print(f"\nBranch: {branch_id}")
    tools = branch.get("tools", [])
    print(f"  Tools: {len(tools)}")
    for tool in tools:
        tool_name = tool.__class__.__name__
        if "MCP" in tool_name:
            print(f"  ✓ Found MCP tool: {tool_name}")
            mcp_tools_found.append(tool_name)
        else:
            print(f"    - {tool_name}")

print("\n" + "=" * 80)
if mcp_tools_found:
    print(f"✅ SUCCESS: Found {len(mcp_tools_found)} MCP tool(s): {', '.join(mcp_tools_found)}")
else:
    print("❌ FAILED: No MCP tools found")
print("=" * 80)

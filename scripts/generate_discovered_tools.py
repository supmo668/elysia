#!/usr/bin/env python3
"""
Generate discovered_tools.yaml - Script to generate tool discovery configuration.
Run this after adding new tools to update the discovery YAML.
"""
import sys
from pathlib import Path

from elysia.util.tool_discovery import generate_tool_discovery_yaml

if __name__ == "__main__":
    output_path = Path.cwd() / "elysia" / "config" / "discovered_tools.yaml"
    yaml_content = generate_tool_discovery_yaml(output_path)
    print(f"Generated tool discovery YAML at: {output_path}")
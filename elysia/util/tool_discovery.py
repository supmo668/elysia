"""
Tool Discovery Utility - Discovers and manages tool metadata for Elysia.

This module provides utilities to:
1. Discover all available tools from custom_tools
2. Generate YAML configurations of discovered tools
3. Provide tool metadata for initialization
"""
import yaml
from pathlib import Path
from typing import Dict, Type, TYPE_CHECKING
from logging import getLogger

if TYPE_CHECKING:
    from elysia import Tool

logger = getLogger(__name__)

GENERATE_TOOL_YAML=True  # Set to True to auto-generate YAML on import

def discover_tools_from_module() -> Dict[str, Type]:
    """
    Discover all Tool subclasses from the custom_tools module and MCP tools.

    Returns:
        Dict[str, Type]: Dictionary mapping tool names to Tool class types
    """
    import elysia.api.custom_tools as custom_tools
    from elysia import Tool

    tool_classes = {}

    # Get all objects from the custom_tools module
    module_objects = dict(
        [
            (name, cls)
            for name, cls in custom_tools.__dict__.items()
            if isinstance(cls, type)
        ]
    )

    # Filter for Tool subclasses (excluding the base Tool class)
    for name, cls in module_objects.items():
        try:
            if issubclass(cls, Tool) and cls.__name__ != "Tool":
                tool_classes[name] = cls
        except TypeError:
            # Not a class, skip
            continue

    # Also discover MCP tools from the mcp module
    try:
        import elysia.tools.mcp.mcp_loader as mcp_loader

        # Get dynamically loaded MCP tool classes
        mcp_module_objects = dict(
            [
                (name, cls)
                for name, cls in mcp_loader.__dict__.items()
                if isinstance(cls, type)
            ]
        )

        # Filter for Tool subclasses
        for name, cls in mcp_module_objects.items():
            try:
                if issubclass(cls, Tool) and cls.__name__ != "Tool" and name not in tool_classes:
                    tool_classes[name] = cls
                    logger.debug(f"Discovered MCP tool: {name}")
            except TypeError:
                continue

    except Exception as e:
        logger.debug(f"Could not discover MCP tools: {e}")

    return tool_classes


def get_tool_metadata(tool_classes: Dict[str, Type] | None = None) -> Dict[str, Dict]:
    """
    Get metadata for all discovered tools.
    
    Args:
        tool_classes: Optional dictionary of tool classes. If None, will discover automatically.
        
    Returns:
        Dict[str, Dict]: Dictionary mapping tool names to their metadata
    """
    if tool_classes is None:
        tool_classes = discover_tools_from_module()
    
    metadata = {}
    for name, cls in tool_classes.items():
        try:
            metadata[name] = cls.get_metadata()
        except Exception as e:
            logger.warning(f"Failed to get metadata for tool {name}: {e}")
            continue
    
    return metadata


def generate_tool_discovery_yaml(output_path: str | Path | None = None) -> str:
    """
    Generate a YAML configuration file of all discovered tools.
    
    Args:
        output_path: Optional path to write the YAML file. If None, returns the YAML string only.
        
    Returns:
        str: The YAML content as a string
    """
    tool_classes = discover_tools_from_module()
    metadata = get_tool_metadata(tool_classes)
    
    # Organize tools by category
    organized_tools = {
        "retrieval": {},
        "text": {},
        "visualization": {},
        "postprocessing": {},
        "mcp": {},
        "other": {}
    }
    
    for tool_name, tool_meta in metadata.items():
        # Determine category based on tool name or metadata
        # Check MCP first to avoid mis-categorization
        if "mcp" in tool_name.lower() or tool_name.startswith("MCP_"):
            category = "mcp"
        elif any(kw in tool_name.lower() for kw in ["query", "aggregate", "search", "retrieve"]):
            category = "retrieval"
        elif any(kw in tool_name.lower() for kw in ["text", "summarize", "summarise", "cite"]):
            category = "text"
        elif any(kw in tool_name.lower() for kw in ["visual", "plot", "chart", "graph", "regression"]):
            category = "visualization"
        elif any(kw in tool_name.lower() for kw in ["summarise_items", "postprocess"]):
            category = "postprocessing"
        else:
            category = "other"
        
        organized_tools[category][tool_name] = {
            "class_name": tool_name,
            "name": tool_meta.get("name", tool_name),
            "description": tool_meta.get("description", "No description available"),
            "end": tool_meta.get("end", False),
            "available": True
        }
    
    # Remove empty categories
    organized_tools = {k: v for k, v in organized_tools.items() if v}
    
    yaml_content = yaml.dump(
        {"discovered_tools": organized_tools},
        default_flow_style=False,
        sort_keys=False,
        width=120
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"Tool discovery YAML written to {output_path}")
    
    return yaml_content


def get_tools_by_category(category: str) -> Dict[str, Type]:
    """
    Get all tools belonging to a specific category.
    
    Args:
        category: Category name (e.g., "retrieval", "text", "visualization")
        
    Returns:
        Dict[str, Type]: Dictionary of tool names to Tool classes in that category
    """
    all_tools = discover_tools_from_module()
    metadata = get_tool_metadata(all_tools)
    
    category_tools = {}
    for tool_name, tool_class in all_tools.items():
        tool_meta = metadata.get(tool_name, {})
        # Simple category detection based on tool name
        if category.lower() in tool_name.lower():
            category_tools[tool_name] = tool_class
    
    return category_tools

# Note: Auto-generation disabled to prevent circular imports at module load time
# Run scripts/generate_discovered_tools.py to manually generate the YAML
# if GENERATE_TOOL_YAML:
#     generate_tool_discovery_yaml()
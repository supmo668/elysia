"""
UI Module - Provides default tool configurations for Tree initialization.

This module centralizes the logic for which tools are loaded by default in different
tree initialization modes (one_branch, multi_branch, etc.).
"""
from typing import TYPE_CHECKING, Dict, List, Tuple, Type
from logging import Logger

from elysia.objects import Tool
from elysia.tools.retrieval.query import Query
from elysia.tools.retrieval.aggregate import Aggregate
from elysia.tools.visualisation.visualise import Visualise
from elysia.tools.postprocessing.summarise_items import SummariseItems
from elysia.tools.text.text import CitedSummarizer, FakeTextResponse

if TYPE_CHECKING:
    from elysia.tree.tree import Tree


# Default tool configurations for different initialization modes
DEFAULT_TOOL_CONFIGS = {
    "multi_branch": {
        "branches": [
            {
                "id": "base",
                "root": True,
                "instruction": """
                Choose a base-level task based on the user's prompt and available information.
                You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
                You can end the conversation by choosing text response, or summarise some retrieved information.
                Base your decision on what information is available and what the user is asking for - you can search multiple times if needed,
                but you should not search if you have already found all the information you need.
                """,
                "status": "Choosing a base-level task...",
                "tools": [
                    {"class": CitedSummarizer},
                    {"class": FakeTextResponse},
                    {"class": Visualise},
                ]
            },
            {
                "id": "search",
                "root": False,
                "from_branch_id": "base",
                "instruction": """
                Choose between querying the knowledge base via semantic/keyword search, or aggregating information by performing operations, on the knowledge base.
                Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query. This is for retrieving specific information via a _query_, similar to a search engine.
                Aggregating is when the user is looking for a specific operations on the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups.
                """,
                "description": """
                Search the knowledge base. This should be used when the user is lacking information for this particular prompt. This retrieves information only and provides no output to the user except the information.
                Choose to query (semantic or keyword search on a knowledge base), or aggregate information (calculate properties/summary statistics/averages and operations on the knowledge bases).
                """,
                "status": "Searching the knowledge base...",
                "tools": [
                    {"class": Query, "kwargs": {"summariser_in_tree": True}},
                    {"class": Aggregate},
                ]
            }
        ],
        "additional_tools": [
            {"class": SummariseItems, "branch_id": "search", "from_tool_ids": ["query"]}
        ]
    },
    "one_branch": {
        "branches": [
            {
                "id": "base",
                "root": True,
                "instruction": """
                Choose a base-level task based on the user's prompt and available information.
                Decide based on the tools you have available as well as their descriptions.
                Read them thoroughly and match the actions to the user prompt.
                """,
                "status": "Choosing a base-level task...",
                "tools": [
                    {"class": CitedSummarizer},
                    {"class": FakeTextResponse},
                    {"class": Aggregate},
                    {"class": Query, "kwargs": {"summariser_in_tree": True}},
                    {"class": Visualise},
                ]
            }
        ],
        "additional_tools": [
            {"class": SummariseItems, "branch_id": "base", "from_tool_ids": ["query"]}
        ]
    },
    "empty": {
        "branches": [
            {
                "id": "base",
                "root": True,
                "instruction": """
                Choose a base-level task based on the user's prompt and available information.
                Decide based on the tools you have available as well as their descriptions.
                Read them thoroughly and match the actions to the user prompt.
                """,
                "status": "Choosing a base-level task...",
                "tools": []
            }
        ],
        "additional_tools": []
    }
}


def load_default_tools_for_mode(
    tree: "Tree",
    mode: str,
    additional_tool_classes: List[Type[Tool]] | None = None,
    logger: Logger | None = None
) -> None:
    """
    Load default tools for a given initialization mode into a Tree.

    Args:
        tree: The Tree instance to load tools into
        mode: The initialization mode ("multi_branch", "one_branch", "empty", or "default")
        additional_tool_classes: Optional list of additional tool classes to add to the base branch
        logger: Optional logger for debugging
    """
    if mode == "default" or mode == "":
        mode = "one_branch"

    if mode not in DEFAULT_TOOL_CONFIGS:
        if logger:
            logger.warning(f"Unknown initialization mode '{mode}', falling back to 'one_branch'")
        mode = "one_branch"

    config = DEFAULT_TOOL_CONFIGS[mode]
    
    if logger:
        logger.info(f"Loading default tools for mode: '{mode}'")
        logger.info(f"Configuration: {len(config['branches'])} branches, {len(config.get('additional_tools', []))} additional tools")

    # Add branches and their tools
    for branch_config in config["branches"]:
        # Add the branch
        if branch_config["root"]:
            tree.add_branch(
                root=True,
                branch_id=branch_config["id"],
                instruction=branch_config["instruction"],
                status=branch_config.get("status", ""),
            )
            if logger:
                logger.info(f"  ✓ Added root branch: '{branch_config['id']}'")
        else:
            tree.add_branch(
                root=False,
                branch_id=branch_config["id"],
                from_branch_id=branch_config.get("from_branch_id", ""),
                instruction=branch_config["instruction"],
                description=branch_config.get("description", ""),
                status=branch_config.get("status", ""),
            )
            if logger:
                logger.info(f"  ✓ Added branch: '{branch_config['id']}' (from '{branch_config.get('from_branch_id')}')")

        # Add tools to the branch
        for tool_config in branch_config["tools"]:
            tool_class = tool_config["class"]
            kwargs = tool_config.get("kwargs", {})
            tree.add_tool(branch_id=branch_config["id"], tool=tool_class, **kwargs)
            if logger:
                logger.info(f"    → Tool: {tool_class.__name__} (branch: {branch_config['id']})")

    # Add additional tools (tools that depend on other tools)
    if logger:
        logger.info(f"Adding {len(config.get('additional_tools', []))} additional tools...")
    for tool_config in config.get("additional_tools", []):
        tree.add_tool(
            tool=tool_config["class"],
            branch_id=tool_config.get("branch_id"),
            from_tool_ids=tool_config.get("from_tool_ids", [])
        )
        if logger:
            logger.info(f"  ✓ Additional tool: {tool_config['class'].__name__} (branch: {tool_config.get('branch_id')}, depends on: {tool_config.get('from_tool_ids', [])})")

    # Auto-discover and add MCP tools to the root branch
    if logger:
        logger.info("Auto-discovering MCP tools...")
    try:
        from elysia.util.tool_discovery import discover_tools_from_module
        
        all_tools = discover_tools_from_module()
        mcp_tools = {
            name: cls for name, cls in all_tools.items()
            if name.startswith("MCP_")  # MCP tools follow naming convention MCP_<servername>
        }

        if mcp_tools:
            base_branch_id = config["branches"][0]["id"]  # Add to root branch
            if logger:
                logger.info(f"Found {len(mcp_tools)} MCP tools to load")
            for tool_name, tool_class in mcp_tools.items():
                try:
                    tree.add_tool(branch_id=base_branch_id, tool=tool_class)
                    if logger:
                        logger.info(f"  ✓ MCP tool: {tool_name} → branch '{base_branch_id}'")
                except Exception as e:
                    if logger:
                        logger.warning(f"  ✗ Failed to load MCP tool {tool_name}: {e}")
        else:
            if logger:
                logger.info("No MCP tools found to auto-load")
    except Exception as e:
        if logger:
            logger.warning(f"Could not auto-discover MCP tools: {e}")

    # Add any extra tool classes provided (e.g., custom tools passed in)
    if additional_tool_classes:
        base_branch_id = config["branches"][0]["id"]  # Add to the first/root branch
        for tool_class in additional_tool_classes:
            try:
                tree.add_tool(branch_id=base_branch_id, tool=tool_class)
                if logger:
                    logger.info(f"Added additional tool {tool_class.__name__} to branch '{base_branch_id}'")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to add tool {tool_class.__name__}: {e}")


def get_available_modes() -> List[str]:
    """
    Get a list of available initialization modes.
    
    Returns:
        List of mode names
    """
    return list(DEFAULT_TOOL_CONFIGS.keys())


def get_mode_description(mode: str) -> str:
    """
    Get a description of what tools are included in a specific mode.
    
    Args:
        mode: The initialization mode
        
    Returns:
        A string describing the mode's tool configuration
    """
    if mode not in DEFAULT_TOOL_CONFIGS:
        return "Unknown mode"
    
    config = DEFAULT_TOOL_CONFIGS[mode]
    branches = [b["id"] for b in config["branches"]]
    tool_count = sum(len(b["tools"]) for b in config["branches"]) + len(config.get("additional_tools", []))
    
    return f"Mode '{mode}' has {len(branches)} branch(es) with {tool_count} total tools"


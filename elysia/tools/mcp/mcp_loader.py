"""
MCP Loader - Automatically loads MCP servers from mcp.json as Elysia Tools.
Supports both stdio and SSE transport types.

Operating Modes (controlled by MCP_AS_AGENT env var):
- Agent Mode (MCP_AS_AGENT=True): Creates one MCPTool (agent gateway) per MCP server
- Individual Mode (MCP_AS_AGENT=False): Creates individual Tool wrappers for each tool in MCP server
"""
import json
import os
import asyncio
from pathlib import Path
from logging import getLogger
from typing import Any, AsyncGenerator

from elysia.tools.mcp.mcp_tool import MCPTool
from elysia.objects import Tool, Status, Error, Text, Result
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.deps.mcp_health import get_health_checker
import dspy

logger = getLogger(__name__)


def load_mcp_servers_from_config() -> list[type[MCPTool]] | list[type[Tool]]:
    """
    Load MCP servers from mcp.json at repository root.
    
    Behavior depends on MCP_AS_AGENT environment variable:
    - True: Each enabled server becomes one MCPTool (agent gateway) class
    - False: Each tool in each server becomes an individual Tool wrapper class
    
    Supports two transport types:
    - stdio: Runs a local MCP server script
    - sse: Connects to a remote MCP server via Server-Sent Events
    """
    # Look for mcp.json at repository root (4 levels up from this file)
    # Path: elysia/tools/mcp/mcp_loader.py -> repo_root
    config_path = Path(__file__).parent.parent.parent.parent / "mcp.json"
    
    if not config_path.exists():
        logger.info(f"No mcp.json found at {config_path}, skipping MCP server loading")
        return []
    
    # Check agent mode from environment
    agent_mode = os.getenv("MCP_AS_AGENT", "True").lower() == "true"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        servers = config.get("servers", [])
        
        # Filter enabled servers
        enabled_servers = [s for s in servers if s.get("enabled", False)]
        
        if not enabled_servers:
            logger.info("No enabled MCP servers found in mcp.json")
            return []
        
        # Perform health checks on all enabled servers
        logger.info(f"Performing health checks on {len(enabled_servers)} MCP server(s)...")
        health_checker = get_health_checker()
        
        # Run health checks synchronously (we're at module import time)
        loop = asyncio.new_event_loop()
        try:
            health_status = loop.run_until_complete(
                health_checker.check_multiple(enabled_servers)
            )
        finally:
            loop.close()
        
        # Filter servers based on health status
        healthy_servers = []
        for server in enabled_servers:
            server_name = server.get("name", "unknown")
            if health_status.get(server_name, False):
                logger.info(f"✓ MCP server '{server_name}' is healthy")
                healthy_servers.append(server)
            else:
                logger.warning(f"✗ MCP server '{server_name}' is not responding - skipping")
        
        if not healthy_servers:
            logger.warning("No healthy MCP servers found, skipping tool creation")
            return []
        
        tool_classes = []
        
        if agent_mode:
            # Agent mode: Create one MCPTool per server
            logger.info("MCP_AS_AGENT=True: Creating agent gateway tools")
            for server_config in healthy_servers:
                server_name = server_config.get("name", "unknown")
                server_description = server_config.get("description", "")
                url = server_config.get("url")
                
                if not url:
                    logger.warning(f"MCP server '{server_name}' missing url")
                    continue
                
                headers = server_config.get("headers", {})
                logger.info(f"Creating agent tool for MCP server: {server_name}")
                tool_class = _create_mcp_tool_class(
                    server_name=server_name,
                    server_description=server_description,
                    url=url,
                    headers=headers
                )
                
                tool_classes.append(tool_class)
                logger.info(f"Created agent tool class: {tool_class.__name__}")
        
        else:
            # Individual mode: Create separate Tool for each MCP server tool
            logger.info("MCP_AS_AGENT=False: Creating individual tool wrappers")
            for server_config in healthy_servers:
                server_name = server_config.get("name", "unknown")
                server_description = server_config.get("description", "")
                url = server_config.get("url")
                
                if not url:
                    logger.warning(f"MCP server '{server_name}' missing url")
                    continue
                
                server_kwargs = {
                    "server_name": server_name,
                    "server_description": server_description,
                    "url": url,
                    "headers": server_config.get("headers", {}),
                }
                
                logger.info(f"Discovering individual tools from MCP server: {server_name}")
                
                # Create individual tool wrappers for this server
                individual_tools = _create_individual_tool_wrappers(**server_kwargs)
                tool_classes.extend(individual_tools)
                logger.info(f"Created {len(individual_tools)} individual tool classes from {server_name}")
        
        return tool_classes
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid mcp.json format: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading MCP configuration: {e}")
        return []


def _create_mcp_tool_class(
    server_name: str,
    server_description: str,
    url: str,
    headers: dict | None = None
) -> type[MCPTool]:
    """
    Dynamically create MCPTool class for an MCP server.
    """
    class_name = f"MCP_{server_name}".replace("-", "_").replace(" ", "_")
    
    _server_name = server_name
    _server_description = server_description
    _url = url
    _headers = headers or {}
    
    class DynamicMCPServer(MCPTool):
        """Dynamically created MCP server tool."""
        
        def __init__(self, **kwargs):
            super().__init__(
                server_name=_server_name,
                server_description=_server_description,
                url=_url,
                headers=_headers,
                **kwargs
            )
    
    # Set the class name
    DynamicMCPServer.__name__ = class_name
    DynamicMCPServer.__qualname__ = class_name
    
    return DynamicMCPServer


def _create_individual_tool_wrappers(
    server_name: str,
    server_description: str,
    url: str,
    headers: dict | None = None
) -> list[type[Tool]]:
    """
    Create individual Tool wrappers for each tool in MCP server.
    Used when MCP_AS_AGENT=False.
    """
    import asyncio
    
    try:
        init_kwargs = {
            "server_name": server_name,
            "server_description": server_description,
            "url": url,
            "headers": headers or {},
        }
        
        # Create temporary MCPTool to discover tools
        temp_tool = MCPTool(**init_kwargs)
        
        # Initialize synchronously to discover tools
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use run_until_complete
            logger.warning(f"Cannot discover individual tools from {server_name} - already in async context")
            return []
        else:
            success = loop.run_until_complete(temp_tool.initialize())
        
        if not success or not temp_tool._langchain_tools:
            logger.warning(f"Failed to discover tools from MCP server: {server_name}")
            return []
        
        # Create a wrapper class for each discovered tool
        tool_classes = []
        for lc_tool in temp_tool._langchain_tools:
            tool_name = getattr(lc_tool, "name", "unknown")
            tool_desc = getattr(lc_tool, "description", "No description")
            
            # Create wrapper class
            wrapper_class = _create_individual_tool_class(
                server_name=server_name,
                url=url,
                headers=headers,
                tool_name=tool_name,
                tool_description=tool_desc,
                langchain_tool=lc_tool
            )
            
            tool_classes.append(wrapper_class)
            logger.debug(f"Created individual tool wrapper: {wrapper_class.__name__}")
        
        return tool_classes
        
    except Exception as e:
        logger.error(f"Error creating individual tool wrappers for {server_name}: {e}")
        return []


def _create_individual_tool_class(
    server_name: str,
    url: str,
    tool_name: str,
    tool_description: str,
    langchain_tool: Any,
    headers: dict | None = None,
) -> type[Tool]:
    """
    Create individual Tool wrapper for specific MCP server tool.
    """
    class_name = f"MCP_{server_name}_{tool_name}".replace("-", "_").replace(" ", "_")
    
    _server_name = server_name
    _tool_name = tool_name
    _tool_description = tool_description
    _url = url
    _headers = headers or {}
    
    # Extract input schema from langchain tool
    tool_inputs = {}
    if hasattr(langchain_tool, "args_schema") and langchain_tool.args_schema:
        schema = langchain_tool.args_schema
        if hasattr(schema, "schema"):
            properties = schema.schema().get("properties", {})
            required = schema.schema().get("required", [])
            
            for prop_name, prop_info in properties.items():
                tool_inputs[prop_name] = {
                    "description": prop_info.get("description", ""),
                    "type": str,  # Simplified type mapping
                    "required": prop_name in required,
                }
    
    class IndividualMCPToolWrapper(Tool):
        """Dynamically created wrapper for individual MCP server tool."""
        
        def __init__(self, **kwargs):
            super().__init__(
                name=f"mcp_{_server_name}_{_tool_name}",
                description=_tool_description,
                status=f"Executing {_tool_name} on MCP server {_server_name}...",
                inputs=tool_inputs if tool_inputs else {},
                end=False,
                **kwargs
            )
            
            self._mcp_connection = None
            self._langchain_tool = None
        
        async def _ensure_connection(self):
            """Connect to MCP server and load tool."""
            if self._langchain_tool is not None:
                return True
            
            try:
                from langchain_mcp_adapters.tools import load_mcp_tools
                from mcp import ClientSession
                from mcp.client.sse import sse_client
                
                async with sse_client(_url, headers=_headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        
                        for tool in tools:
                            if getattr(tool, "name", "") == _tool_name:
                                self._langchain_tool = tool
                                return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server for {_tool_name}: {e}")
                return False
        
        async def __call__(
            self,
            tree_data: TreeData,
            inputs: dict,
            base_lm: dspy.LM,
            complex_lm: dspy.LM,
            client_manager: ClientManager | None = None,
            **kwargs,
        ) -> AsyncGenerator[Result | Status | Error | Text, None]:
            """Execute the specific MCP tool."""
            yield Status(f"Executing {_tool_name} on MCP server {_server_name}...")
            
            # Ensure connection
            if not await self._ensure_connection():
                yield Error(error_message=f"Failed to connect to MCP server {_server_name}")
                return
            
            try:
                # Execute the tool
                if hasattr(self._langchain_tool, "ainvoke"):
                    result = await self._langchain_tool.ainvoke(inputs)
                else:
                    result = self._langchain_tool.invoke(inputs)
                
                # Process result
                if isinstance(result, str):
                    yield Text(result)
                elif isinstance(result, dict):
                    yield Result([result], name=_tool_name)
                elif isinstance(result, list):
                    yield Result(
                        [item if isinstance(item, dict) else {"value": item} for item in result],
                        name=_tool_name,
                    )
                else:
                    yield Result([{"result": str(result)}], name=_tool_name)
                    
            except Exception as e:
                error_msg = f"Error executing {_tool_name}: {str(e)}"
                logger.error(error_msg)
                yield Error(error_message=error_msg)
        
        async def is_tool_available(
            self,
            tree_data: TreeData,
            base_lm: dspy.LM,
            complex_lm: dspy.LM,
            client_manager: ClientManager,
        ) -> bool:
            """Check if tool is available."""
            return True
    
    # Set the class name
    IndividualMCPToolWrapper.__name__ = class_name
    IndividualMCPToolWrapper.__qualname__ = class_name
    
    return IndividualMCPToolWrapper


# Load servers at module import time [IMPORTANT]
_loaded_tool_classes = load_mcp_servers_from_config()

# Export to module namespace for discovery
for tool_class in _loaded_tool_classes:
    globals()[tool_class.__name__] = tool_class

__all__ = [tool_class.__name__ for tool_class in _loaded_tool_classes]

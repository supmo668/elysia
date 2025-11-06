"""  
MCP Tool - Elysia Tool for MCP server integration via SSE/HTTP.

Connects to running MCP servers over HTTP using Server-Sent Events.

Operating Modes (controlled by MCP_AS_AGENT env var):
- Agent Mode (True): Natural language queries via ReAct agent
- Gateway Mode (False): Direct tool execution with structured inputs
"""
from __future__ import annotations
from typing import Any, AsyncGenerator, TYPE_CHECKING
from logging import Logger, getLogger
import os

import dspy

from elysia.objects import Tool, Status, Error, Text, Result
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
if TYPE_CHECKING:
    from elysia.tree.tree import Tree

logger = getLogger(__name__)


class MCPTool(Tool):
    """
    Elysia Tool that connects to a running MCP server via SSE/HTTP.
    
    Usage:
        tool = MCPTool(
            server_name="api_server",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token"},
            server_description="MCP server with AI tools"
        )
    """

    def __init__(
        self,
        server_name: str,
        url: str,
        headers: dict[str, str] | None = None,
        server_description: str | None = None,
        logger: Logger | None = None,
        **kwargs,
    ):
        """
        Args:
            server_name: Name identifier for this MCP server
            url: URL for SSE endpoint (e.g., "http://localhost:8080/mcp")
            headers: HTTP headers for SSE connection
            server_description: Custom description for the MCP server
            logger: Optional logger
        """
        self.server_name = server_name
        self.url = url
        self.headers = headers or {}
        self.logger = logger or globals()['logger']
        self._langchain_tools: list[Any] = []
        self._initialized = False
        self._agent = None
        
        # Check MCP_AS_AGENT mode from environment
        self.agent_mode = os.getenv("MCP_AS_AGENT", "True").lower() == "true"

        if not url:
            raise ValueError("url is required for SSE transport")

        # Use custom description if provided, otherwise use generic one
        if server_description:
            description = server_description
        else:
            if self.agent_mode:
                description = f"MCP server '{server_name}' - AI agent with access to multiple tools. Accepts natural language queries."
            else:
                description = f"MCP server '{server_name}' - provides access to multiple tools via Model Context Protocol"

        # Define inputs based on mode
        if self.agent_mode:
            # Agent mode: simple natural language query input
            inputs = {
                "query": {
                    "description": "Natural language query or task description for the AI agent to execute using available MCP tools",
                    "type": str,
                    "required": True,
                }
            }
        else:
            # Gateway mode: action/tool_name/tool_inputs pattern
            inputs = {
                "action": {
                    "description": "Action: 'list' to show tools, 'execute' to run a specific tool",
                    "type": str,
                    "default": "list",
                },
                "tool_name": {
                    "description": "Name of the tool to execute (required when action='execute')",
                    "type": str,
                    "required": False,
                },
                "tool_inputs": {
                    "description": "Inputs for the tool (required when action='execute')",
                    "type": dict,
                    "required": False,
                    "default": {},
                },
            }

        super().__init__(
            name=f"mcp_{server_name}",
            description=description,
            status="Ready to connect to MCP server",
            inputs=inputs,
            end=False,
            **kwargs,
        )

    async def initialize(self) -> bool:
        """Connect to MCP server and discover available tools."""
        if self._initialized:
            return True

        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            
            if self.logger:
                self.logger.info(f"Connecting to MCP server: {self.server_name} at {self.url}")

            async with sse_client(self.url, headers=self.headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._langchain_tools = await load_mcp_tools(session)

            self._initialized = True
            
            # Initialize agent if in agent mode
            if self.agent_mode:
                try:
                    from langchain.agents import create_react_agent, AgentExecutor
                    from langchain_core.prompts import PromptTemplate
                    from langchain_openai import ChatOpenAI
                    
                    # Create LangChain LLM (you can make this configurable)
                    llm = ChatOpenAI(temperature=0, model=os.getenv("BASE_MODEL", "gpt-4o-mini"))
                    
                    # Create ReAct agent with MCP tools
                    prompt = PromptTemplate.from_template(
                        """Answer the following question as best you can. You have access to the following tools:

                        {tools}

                        Use the following format:

                        Question: the input question you must answer
                        Thought: you should always think about what to do
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: the final answer to the original input question

                        Begin!

                        Question: {input}
                        Thought:{agent_scratchpad}"""
                    )
                    
                    agent = create_react_agent(llm, self._langchain_tools, prompt)
                    self._agent = AgentExecutor(agent=agent, tools=self._langchain_tools, verbose=True, handle_parsing_errors=True)
                    
                    if self.logger:
                        self.logger.info(f"Created ReAct agent with {len(self._langchain_tools)} tools from {self.server_name}")
                except ImportError as e:
                    if self.logger:
                        self.logger.error(f"Failed to create agent: {e}")
                        self.logger.error("Install with: pip install langchain langchain-openai")
                    return False
            
            if self.logger:
                self.logger.info(f"Loaded {len(self._langchain_tools)} tools from {self.server_name}")

            return True

        except ImportError as e:
            if self.logger:
                self.logger.error(f"langchain-mcp-adapters not installed: {e}")
                self.logger.error("Install with: pip install langchain-mcp-adapters")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize MCP server '{self.server_name}': {str(e)}")
            return False

    def get_tool_by_name(self, tool_name: str) -> Any | None:
        """Get a specific LangChain tool by name."""
        for tool in self._langchain_tools:
            if getattr(tool, "name", "") == tool_name:
                return tool
        return None
 
    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ) -> AsyncGenerator[Result | Status | Error | Text, None]:
        """Execute in agent mode (natural language query) or gateway mode (action-based)."""
        # Initialize if needed
        if not self._initialized:
            yield Status(f"Initializing MCP server '{self.server_name}'...")
            success = await self.initialize()
            if not success:
                yield Error(error_message=f"Failed to initialize MCP server '{self.server_name}'")
                return
        
        # Handle based on mode
        if self.agent_mode:
            # Agent mode: execute natural language query with ReAct agent
            query = inputs.get("query", "")
            
            if not query:
                yield Error(error_message="Query is required in agent mode")
                return
            
            if not self._agent:
                yield Error(error_message="Agent not initialized. Check logs for initialization errors.")
                return
            
            yield Status(f"Executing query with AI agent on '{self.server_name}': {query}")
            
            try:
                # Execute agent
                result = await self._agent.ainvoke({"input": query})
                
                if self.logger:
                    self.logger.debug(f"Agent result: {result}")
                
                # Extract output
                output = result.get("output", str(result))
                yield Text(str(output))
                
            except Exception as e:
                error_msg = f"Error executing agent on '{self.server_name}': {str(e)}"
                if self.logger:
                    self.logger.error(error_msg)
                yield Error(error_message=error_msg)
            
            return
        
        # Gateway mode: traditional action-based execution
        action = inputs.get("action", "list")

        # List available tools
        if action == "list":
            tool_info = [
                {
                    "name": getattr(t, "name", "unknown"),
                    "description": getattr(t, "description", "No description"),
                }
                for t in self._langchain_tools
            ]
            yield Result(tool_info, name=f"{self.server_name}_tools")
            yield Text(
                f"MCP Server '{self.server_name}' has {len(self._langchain_tools)} tools:\n"
                + "\n".join([f"- {t['name']}: {t['description']}" for t in tool_info])
            )

        # Execute a specific tool
        elif action == "execute":
            tool_name = inputs.get("tool_name")
            tool_inputs = inputs.get("tool_inputs", {})

            if not tool_name:
                yield Error(error_message="tool_name is required when action='execute'")
                return

            # Find the tool
            lc_tool = self.get_tool_by_name(tool_name)
            if not lc_tool:
                available = [getattr(t, "name", "") for t in self._langchain_tools]
                yield Error(
                    error_message=f"Tool '{tool_name}' not found. Available: {available}"
                )
                return

            # Execute the tool
            yield Status(f"Executing '{tool_name}' on MCP server '{self.server_name}'...")

            try:
                if hasattr(lc_tool, "ainvoke"):
                    result = await lc_tool.ainvoke(tool_inputs)
                else:
                    result = lc_tool.invoke(tool_inputs)

                if self.logger:
                    self.logger.debug(f"Tool '{tool_name}' result: {result}")

                # Process result
                if isinstance(result, str):
                    yield Text(result)
                elif isinstance(result, dict):
                    yield Result([result], name=tool_name)
                elif isinstance(result, list):
                    yield Result(
                        [item if isinstance(item, dict) else {"value": item} for item in result],
                        name=tool_name,
                    )
                else:
                    yield Result([{"result": str(result)}], name=tool_name)

            except Exception as e:
                error_msg = f"Error executing '{tool_name}' on '{self.server_name}': {str(e)}"
                if self.logger:
                    self.logger.error(error_msg)
                yield Error(error_message=error_msg)

        else:
            yield Error(error_message=f"Unknown action '{action}'. Use 'list' or 'execute'")

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
    ) -> bool:
        """Check if the MCP server is available."""
        return bool(self.url)


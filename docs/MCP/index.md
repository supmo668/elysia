# MCP (Model Context Protocol) Integration

Complete documentation for integrating MCP servers with Elysia.

## Quick Links

- **[MCP_AS_AGENT Configuration](mcp_as_agent.md)** - ⭐ **NEW**: Control MCP behavior (Agent vs Individual mode)
- **[Health Checks](health_checks.md)** - ⭐ **NEW**: Automatic MCP server health validation
- **[Overview](overview.md)** - High-level summary of MCP integration
- **[Quickstart](quickstart.md)** - 30-second setup guide
- **[Interaction Model](interaction_model.md)** - How MCPTool works and parameter surfacing
- **[Architecture](architecture.md)** - Technical architecture details
- **[API Integration](api_integration.md)** - Complete API flow documentation
- **[Usage Guide](usage_guide.md)** - Configuration and usage instructions
- **[Implementation Details](implementation_details.md)** - Detailed changes, diagrams, and testing

## Documentation Structure

### For Users

1. **Start with [Overview](overview.md)** - Understand what was built
2. **Read [Quickstart](quickstart.md)** - Get up and running quickly
3. **Learn [Interaction Model](interaction_model.md)** - Understand how MCPTool works
4. **Review [Usage Guide](usage_guide.md)** - Learn configuration options

### For Developers

1. **Study [Interaction Model](interaction_model.md)** - Understand the gateway pattern
2. **Study [Architecture](architecture.md)** - Understand the implementation
3. **Review [API Integration](api_integration.md)** - See the complete flow
4. **Read [Implementation Details](implementation_details.md)** - See code changes and testing
5. **Refer to source code** in `elysia/tools/mcp/`

## Key Concepts

### Operating Modes (NEW)

Elysia supports two modes for MCP integration controlled by `MCP_AS_AGENT` environment variable:

1. **Agent Mode** (`MCP_AS_AGENT=True`) - Default
   - Natural language interface  
   - One AI agent tool per MCP server
   - Autonomous task execution via ReAct agent
   - Example: `tree("Search for ML papers and analyze them")`

2. **Individual Tool Mode** (`MCP_AS_AGENT=False`)
   - Structured parameter interface
   - One Elysia tool per MCP server tool
   - Direct tool invocation
   - Example: `tree.tools["mcp_server_search"](inputs={...})`

See [MCP_AS_AGENT Configuration](mcp_as_agent.md) for complete details.

### Gateway Pattern

Each `MCPTool` represents **one MCP server** that provides access to **multiple tools**:

- **Not**: Each MCP tool as a separate Elysia tool ❌
- **Yes**: One MCPTool per MCP server, exposing tools via actions ✅

See [Interaction Model](interaction_model.md) for detailed explanation.

### Two-Phase Operation

1. **Discovery**: `action='list'` - Discover available tools and their schemas
2. **Execution**: `action='execute'` - Execute a specific tool by name

### Parameter Surfacing

- **Level 1 (Elysia)**: `action`, `tool_name`, `tool_inputs` (always visible)
- **Level 2 (MCP)**: Each tool's specific parameters (discovered dynamically)

## Key Features

✅ **Automatic Discovery**: MCP tools are automatically discovered and registered  
✅ **Health Checks**: Servers validated before initialization (NEW)  
✅ **API Integration**: Full integration with Elysia's UserManager and TreeManager  
✅ **Configuration-Based**: Enable/disable servers via `mcp.json`  
✅ **Production Ready**: Error handling, logging, type safety  
✅ **Minimal Code**: ~200 lines per component, no overengineering  

## Installation

The MCP integration uses `langchain-mcp-adapters`:

```bash
pip install langchain-mcp-adapters
```

This is already included in Elysia's dependencies.

## Current Status

- ✅ Integration complete and functional
- ✅ API flow tested and verified
- ✅ Documentation comprehensive
- ⏳ No MCP servers currently configured
- 🎯 Ready for production use

## Getting Help

- **Configuration Issues**: See [Usage Guide](usage_guide.md#troubleshooting)
- **Technical Details**: See [Architecture](architecture.md)
- **API Flow**: See [API Integration](api_integration.md)

---

**MCP integration is ready to use! Configure `mcp.json` to enable servers.** 🚀


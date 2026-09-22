"""
Application-level tool registry for raj-ai-agentkit tool execution demo.

Registry owns tool registration, lookup, parameter mapping, and safe execution.
Maintained entirely within application code.
"""

from typing import Any, Callable, Dict, Optional
from tools import calculate, system_info, lookup_knowledge, failing_tool

# Application tool registry dictionary
TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "calculator": calculate,
    "system_info": system_info,
    "knowledge_lookup": lookup_knowledge,
    "failing_tool": failing_tool,
}


def get_tool(tool_name: str) -> Optional[Callable[..., Dict[str, Any]]]:
    """Returns tool function if registered, otherwise None."""
    return TOOLS.get(tool_name)


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Safely executes a registered tool by name with provided arguments.
    Captures tool exceptions cleanly without crashing the host application.
    """
    if tool_name not in TOOLS:
        return {
            "status": "error",
            "error": f"Unknown tool '{tool_name}'. Available tools: {list(TOOLS.keys())}",
            "tool_name": tool_name,
        }

    tool_func = TOOLS[tool_name]
    try:
        result = tool_func(**kwargs)
        return result
    except Exception as err:
        return {
            "status": "error",
            "error": f"Tool '{tool_name}' execution failed: {type(err).__name__} - {str(err)}",
            "tool_name": tool_name,
        }

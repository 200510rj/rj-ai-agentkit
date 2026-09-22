"""
Application-level deterministic tools for the raj-ai-agentkit tool execution demo.

These tools are maintained entirely within application code outside src/agentkit/.
"""

import ast
import operator
import platform
import sys
import os
from typing import Any, Dict

# Import existing application retriever
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_demo"))
from embedding_retriever import EmbeddingRetriever

# Initialize application retriever instance
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "rag_demo", "knowledge")
_retriever = EmbeddingRetriever(knowledge_dir=KNOWLEDGE_PATH)


# Supported AST operators for safe arithmetic calculation
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluates safe mathematical AST nodes only."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # Python < 3.8 compatibility
        return float(node.n)
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsafe or invalid AST node: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely calculates arithmetic expressions without using arbitrary eval().
    Supports +, -, *, /, //, %, **.
    """
    if not expression or not isinstance(expression, str):
        return {"status": "error", "error": "Invalid input expression"}

    # Sanitize & parse AST
    cleaned_expr = expression.strip()
    try:
        parsed_ast = ast.parse(cleaned_expr, mode="eval")
        result = _safe_eval(parsed_ast)
        # Return int if whole number
        if result.is_integer():
            result = int(result)
        return {"status": "success", "result": result, "expression": cleaned_expr}
    except Exception as err:
        return {"status": "error", "error": f"Calculation error: {str(err)}", "expression": cleaned_expr}


def system_info() -> Dict[str, Any]:
    """
    Returns safe local system information.
    Does NOT expose environment variables, API keys, passwords, or tokens.
    """
    return {
        "status": "success",
        "platform": platform.platform(),
        "system": platform.system(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "Unknown",
    }


def lookup_knowledge(query: str, top_k: int = 2) -> Dict[str, Any]:
    """
    Executes knowledge retrieval via application-level EmbeddingRetriever.
    """
    try:
        chunks = _retriever.retrieve(query, top_k=top_k, similarity_threshold=0.50)
        return {
            "status": "success",
            "query": query,
            "chunks": chunks,
            "count": len(chunks),
        }
    except Exception as err:
        return {"status": "error", "error": f"Retrieval failed: {str(err)}"}


def failing_tool() -> Dict[str, Any]:
    """Tool that intentionally raises an exception to verify error handling."""
    raise RuntimeError("intentional test failure")

# server.py
import ast
import math
import operator
import random
import sys
import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger('Calculator')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# Create an MCP server
mcp = FastMCP("Calculator")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _preview_expression(expr: str, max_chars: int = 160) -> str:
    compact = ' '.join((expr or '').split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + '...'


def _safe_eval(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")
    modules = {"math": math, "random": random}

    def eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return _BIN_OPS[op_type](eval_node(node.left), eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARY_OPS[op_type](eval_node(node.operand))

        if isinstance(node, ast.Attribute):
            value = eval_node(node.value)
            if value in modules.values() and not node.attr.startswith("__"):
                attr = getattr(value, node.attr, None)
                if attr is None:
                    raise ValueError(f"Unknown attribute: {node.attr}")
                return attr
            raise ValueError("Attribute access is restricted")

        if isinstance(node, ast.Name):
            if node.id in modules:
                return modules[node.id]
            raise ValueError(f"Unknown symbol: {node.id}")

        if isinstance(node, ast.Call):
            func = eval_node(node.func)
            args = [eval_node(arg) for arg in node.args]
            if not callable(func):
                raise ValueError("Target is not callable")
            return func(*args)

        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    return eval_node(tree)

# Add an addition tool
@mcp.tool()
def calculator(python_expression: str) -> dict:
    """For mathamatical calculation, always use this tool to calculate the result of a python expression. You can use 'math' or 'random' directly, without 'import'."""
    logger.info('Tool calculator invoked expression=%s', _preview_expression(python_expression))
    try:
        result = _safe_eval(python_expression)
    except Exception as exc:
        logger.warning("Tool calculator failed: %s", exc)
        return {"success": False, "error": str(exc)}

    logger.info('Tool calculator completed success=true result=%s', result)
    return {"success": True, "result": result}

# Start the server
if __name__ == "__main__":
    mcp.run(transport="stdio")

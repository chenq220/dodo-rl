"""AutoML 业务指标：标准 infos/metadata 提取和受限算术表达式求值。"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Iterable, Mapping

from scheduler.model import AutoMLMetricOutputConfig


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"min": min, "max": max, "abs": abs}


def _path_value(root: Mapping[str, Any], path: str, *, label: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"{label} path not found: {path}")
        value = value[part]
    return value


def _number(value: Any, *, label: str) -> float:
    # infos中的 died/line_breached/stage_success 通常是bool，按标准0/1指标处理。
    if not isinstance(value, (bool, int, float)):
        raise ValueError(f"{label} must be a numeric scalar, given={value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, given={value!r}")
    return result


def reduce_values(values: Iterable[float], reducer: str) -> float:
    items = list(values)
    if not items:
        raise ValueError(f"cannot apply {reducer} to an empty metric series")
    if reducer == "sum":
        return sum(items)
    if reducer == "mean":
        return sum(items) / len(items)
    if reducer == "min":
        return min(items)
    if reducer == "max":
        return max(items)
    if reducer == "latest":
        return items[-1]
    raise ValueError(f"unsupported metric reducer: {reducer}")


class EpisodeMetricCollector:
    """按 step 收集 infos，并在局末输出配置声明的逐局标量。"""

    def __init__(self, outputs: Mapping[str, AutoMLMetricOutputConfig | Mapping[str, Any]]):
        self.outputs = {
            name: value if isinstance(value, AutoMLMetricOutputConfig)
            else AutoMLMetricOutputConfig.from_dict(dict(value))
            for name, value in outputs.items()
        }
        self._steps: dict[str, list[float]] = {
            name: [] for name, cfg in self.outputs.items() if cfg.info_path
        }

    def add_infos(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        for name, config in self.outputs.items():
            if not config.info_path:
                continue
            per_agent = [
                _number(
                    _path_value(info, config.info_path, label=f"metric_outputs.{name}.info"),
                    label=f"metric_outputs.{name}",
                )
                for info in infos.values()
            ]
            self._steps[name].append(reduce_values(per_agent, config.agent_reduce))

    def finish(self, env_metadata: Mapping[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, config in self.outputs.items():
            if config.metadata_path:
                result[name] = _number(
                    _path_value(
                        env_metadata,
                        config.metadata_path,
                        label=f"metric_outputs.{name}.metadata",
                    ),
                    label=f"metric_outputs.{name}",
                )
            else:
                result[name] = reduce_values(self._steps[name], config.episode_reduce)
        return result


def _expression_tree(expression: str) -> ast.Expression:
    if not expression.strip():
        raise ValueError("business_metric.expression cannot be empty")
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid business_metric.expression: {exc.msg}") from exc


def validate_business_expression(expression: str, variables: set[str]) -> None:
    tree = _expression_tree(expression)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.Constant)):
            if isinstance(node, ast.Constant) and (
                    isinstance(node.value, bool) or not isinstance(node.value, (int, float))
            ):
                raise ValueError("business_metric.expression only supports numeric constants")
            continue
        if isinstance(node, ast.Name):
            if node.id not in variables and node.id not in _FUNCTIONS:
                raise ValueError(f"unknown business metric output: {node.id}")
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
                raise ValueError("business_metric.expression contains an unsupported function call")
            continue
        if isinstance(node, tuple(_BINARY_OPS) + tuple(_UNARY_OPS)):
            continue
        raise ValueError(f"business_metric.expression contains unsupported syntax: {type(node).__name__}")


def business_expression_names(expression: str) -> set[str]:
    """返回表达式实际引用的指标输出名，不包含允许的函数名。"""
    return {
        node.id for node in ast.walk(_expression_tree(expression))
        if isinstance(node, ast.Name) and node.id not in _FUNCTIONS
    }


def evaluate_business_expression(expression: str, variables: Mapping[str, float]) -> float:
    validate_business_expression(expression, set(variables))
    tree = _expression_tree(expression)

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return _number(variables[node.id], label=node.id)
        if isinstance(node, ast.BinOp):
            return _BINARY_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call):
            return float(_FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args)))
        raise ValueError(f"unsupported business expression node: {type(node).__name__}")

    return _number(evaluate(tree), label="business_metric result")

"""L1 校验原语 — Pydantic 入口模型 + @validate_args 装饰器

两层分工：
- Pydantic model（MCP 外边界）：类型、范围、格式
- @validate_args（ops 内边界）：业务不变量（memory_id > 0）

Issue #11: 三层防御从审查清单升级为框架原语。
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# ── 校验器函数（B: ops 内边界）──


def positive_int(value: Any, name: str = "value") -> int:
    """正整数校验"""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数, got {value!r}")
    return value


def non_empty_str(value: Any, name: str = "value") -> str:
    """非空字符串校验（自动 strip）"""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须为字符串, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} 不能为空")
    return stripped


# ── @validate_args 装饰器 ──


def validate_args(**validators: Callable) -> Callable:
    """ops 层参数校验装饰器

    支持位置参数和关键字参数。校验失败 → logger.error + 返回 None。

    用法::

        @validate_args(memory_id=positive_int)
        def get_memory(conn, memory_id: int) -> dict | None:
    """
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        @wraps(fn)
        def wrapper(*args, **kwargs):
            args_list = list(args)
            for param_name, validator_fn in validators.items():
                # 关键字参数
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator_fn(
                            kwargs[param_name], param_name,
                        )
                    except (ValueError, TypeError) as e:
                        logger.error("L1 %s.%s: %s", fn.__name__, param_name, e)
                        return None
                # 位置参数
                elif param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args_list):
                        try:
                            args_list[idx] = validator_fn(
                                args_list[idx], param_name,
                            )
                        except (ValueError, TypeError) as e:
                            logger.error(
                                "L1 %s.%s: %s", fn.__name__, param_name, e,
                            )
                            return None
            return fn(*args_list, **kwargs)

        return wrapper

    return decorator


# ── Pydantic 入口模型（C: MCP 外边界）──


class LearnParams(BaseModel):
    """learn tool 参数"""

    insight: str = Field(min_length=1, max_length=300)
    context: str = ""
    collection: str | None = None
    session_id: str | None = None

    @field_validator("insight")
    @classmethod
    def strip_insight(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("insight 不能为空白")
        return v


class RecallParams(BaseModel):
    """recall tool 参数"""

    query: str = Field(min_length=1)
    collection: str | None = None
    n: int = Field(default=5, ge=1, le=100)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    session_id: str | None = None
    context_filter: str | None = None  # JSON 字符串，如 '{"task.success": true}'
    spatial_sort: str | None = None    # JSON 字符串，如 '{"field": "spatial.position", "target": [1,2,3]}'

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query 不能为空白")
        return v


class SavePerceptionParams(BaseModel):
    """save_perception tool 参数"""

    description: str = Field(min_length=5)
    perception_type: Literal[
        "visual", "tactile", "auditory", "proprioceptive", "procedural",
    ] = "visual"
    data: str | None = None
    metadata: str | None = None
    collection: str | None = None
    session_id: str | None = None

    @field_validator("description")
    @classmethod
    def strip_description(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("description 至少 5 个字符")
        return v


class ForgetParams(BaseModel):
    """forget tool 参数"""

    memory_id: int = Field(gt=0)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("reason 不能为空白")
        return v


class UpdateParams(BaseModel):
    """update tool 参数"""

    memory_id: int = Field(gt=0)
    new_content: str = Field(min_length=1, max_length=300)
    context: str = ""

    @field_validator("new_content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("new_content 不能为空白")
        return v


class StartSessionParams(BaseModel):
    """start_session tool 参数"""

    collection: str | None = None
    context: str | None = None


class EndSessionParams(BaseModel):
    """end_session tool 参数"""

    session_id: str = Field(min_length=1)
    outcome_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("session_id")
    @classmethod
    def strip_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id 不能为空白")
        return v


def parse_params(model_class: type[BaseModel], **kwargs) -> BaseModel | dict:
    """解析 MCP tool 参数，校验失败返回 error dict

    用法::

        result = parse_params(LearnParams, insight=insight, context=context)
        if isinstance(result, dict):
            return result  # {"error": "..."}
        params = result
    """
    try:
        return model_class(**kwargs)
    except ValidationError as e:
        first_error = e.errors()[0]
        field = first_error.get("loc", ["unknown"])[0]
        msg = first_error.get("msg", str(e))
        return {"error": f"参数校验失败 ({field}): {msg}"}

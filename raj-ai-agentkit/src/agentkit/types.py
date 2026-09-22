"""
Shared Pydantic data models for agentkit engine outputs and component results.
"""

from typing import Any, Callable, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class BinaryResult(BaseModel):
    """Result returned by engine.check()."""
    model_config = ConfigDict(frozen=True)

    result: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=1000)


class ChoiceResult(BaseModel):
    """Result returned by engine.classify()."""
    model_config = ConfigDict(frozen=True)

    selected: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    reasoning: str = Field(default="", max_length=1000)

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Score for '{key}' must be between 0.0 and 1.0, got {val}")
        return v


class ScoreResult(BaseModel):
    """Result returned by engine.score()."""
    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=0.0, le=1.0)
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=1000)

    @field_validator("criteria_scores")
    @classmethod
    def validate_criteria_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Criterion score for '{key}' must be between 0.0 and 1.0, got {val}")
        return v


class Violation(BaseModel):
    """Represents a safety policy violation detected by Guard."""
    model_config = ConfigDict(frozen=True)

    check_name: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    confidence: float = Field(..., ge=0.0, le=1.0)
    detail: str = ""


class GuardResult(BaseModel):
    """User-facing result returned by Guard.check()."""
    model_config = ConfigDict(frozen=True)

    verdict: Literal["allow", "block"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    violations: list[Violation] = Field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteConfig(BaseModel):
    """Configuration for a specific route in Router."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    handler: Callable[[str], Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteResult(BaseModel):
    """User-facing result returned by Router.route()."""
    model_config = ConfigDict(frozen=True)

    verdict: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GateResult(BaseModel):
    """User-facing result returned by RAGGate.should_retrieve()."""
    model_config = ConfigDict(frozen=True)

    should_retrieve: bool
    verdict: Literal["retrieve", "skip"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """User-facing result returned by Judge.score() or Judge.compare()."""
    model_config = ConfigDict(frozen=True)

    verdict: Literal["pass", "fail", "borderline"]
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """SearchResult returned by VectorStore.search()."""
    model_config = ConfigDict(frozen=True)

    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None

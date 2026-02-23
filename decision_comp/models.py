from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, PositiveInt, validator


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class CriterionSchema(StrictBaseModel):
    name: str = Field(..., min_length=1)
    weight: PositiveInt = Field(..., ge=1, le=10, description="Importance weight from 1 (lowest) to 10 (highest).")
    description: Optional[str] = None


class OptionSchema(StrictBaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None


class OptionCriterionScore(StrictBaseModel):
    option_name: str
    criterion_name: str
    score: PositiveInt = Field(..., ge=1, le=10, description="Score from 1 (worst) to 10 (best).")
    justification: str = Field(..., min_length=1)

    @validator("justification")
    def justification_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Justification must not be empty.")
        return v


class DecisionInputState(StrictBaseModel):
    problem_description: str = Field(..., min_length=1)
    options: List[OptionSchema] = Field(..., min_items=2)
    criteria: List[CriterionSchema] = Field(..., min_items=1)


class AIResearchResultState(StrictBaseModel):
    scores: Dict[Tuple[str, str], OptionCriterionScore] = Field(
        default_factory=dict,
        description="Mapping of (option_name, criterion_name) to researched score and justification.",
    )


class FinalScoresState(StrictBaseModel):
    scores: Dict[Tuple[str, str], OptionCriterionScore] = Field(
        default_factory=dict,
        description="Final human-approved scores and justifications per (option, criterion).",
    )


class CriterionContribution(StrictBaseModel):
    criterion_name: str
    weight: int
    score: int
    contribution: int


class OptionWSMResult(StrictBaseModel):
    option_name: str
    total_score: int
    contributions: List[CriterionContribution]


class WSMResult(StrictBaseModel):
    options: List[OptionWSMResult]
    winner: str
    loser: Optional[str] = None


class SynthesisState(StrictBaseModel):
    explanation: str


class GraphState(StrictBaseModel):
    """
    Aggregate state container used by LangGraph.
    Fields are optional so that different nodes can progressively enrich the state.
    """

    inputs: Optional[DecisionInputState] = None
    ai_scores: Optional[AIResearchResultState] = None
    final_scores: Optional[FinalScoresState] = None
    wsm_result: Optional[WSMResult] = None
    explanation: Optional[str] = None


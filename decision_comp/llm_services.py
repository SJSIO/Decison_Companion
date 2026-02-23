from __future__ import annotations

import os
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator

from .models import DecisionInputState, WSMResult


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class OptionCriterionAIResearch(StrictBaseModel):
    """
    AI-proposed score for a specific (option, criterion) pair, referenced by index.
    The indices must correspond exactly to the order of options and criteria
    provided in the decision input.
    """

    option_index: int = Field(..., ge=0, description="Zero-based index into the options list.")
    criterion_index: int = Field(..., ge=0, description="Zero-based index into the criteria list.")
    score: PositiveInt = Field(..., ge=1, le=10, description="Score from 1 (worst) to 10 (best).")
    justification: str = Field(..., min_length=1)

    @field_validator("justification")
    def justification_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Justification must not be empty.")
        return v


class ResearchBatchOutput(StrictBaseModel):
    items: List[OptionCriterionAIResearch]


class SynthesisOutput(StrictBaseModel):
    explanation: str = Field(..., min_length=1)


def get_llm(model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Instantiate a ChatGroq LLM client.
    Raises a clear error if GROQ_API_KEY is not configured.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running the decision companion."
        )

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        max_retries=2,
        timeout=60,
    )


def run_research_llm(decision_input: DecisionInputState) -> ResearchBatchOutput:
    """
    Call the Groq Llama model to research each option against each criterion.
    Returns a strictly validated ResearchBatchOutput.
    """
    llm = get_llm()

    system_prompt = (
        "You are an analytical assistant helping evaluate decision options. "
        "For every combination of option and criterion, you must assign an integer "
        "score from 1 (worst) to 10 (best) and provide a concise, factual, one-sentence justification. "
        "Base your answers only on widely accepted facts and the descriptions provided. "
        "If information is unclear, make a conservative, clearly justified estimate."
    )

    # Flatten options and criteria into an explicit, index-based description
    # that the model can follow without renaming anything.
    options_block_lines = []
    for idx, opt in enumerate(decision_input.options):
        options_block_lines.append(
            f"{idx}: {opt.name} - {opt.description or 'No additional description.'}"
        )
    options_block = "\n".join(options_block_lines)

    criteria_block_lines = []
    for idx, crit in enumerate(decision_input.criteria):
        criteria_block_lines.append(
            f"{idx}: {crit.name} (weight {crit.weight}/10) - "
            f"{crit.description or 'No additional description.'}"
        )
    criteria_block = "\n".join(criteria_block_lines)

    user_prompt = (
        f"Decision problem:\n{decision_input.problem_description}\n\n"
        f"Options (indexed by option_index):\n{options_block}\n\n"
        f"Criteria (indexed by criterion_index):\n{criteria_block}\n\n"
        "You MUST create one item for EVERY possible combination of option_index "
        "and criterion_index.\n"
        "For each item, you MUST output exactly:\n"
        "- option_index (integer, matching one of the listed option indices)\n"
        "- criterion_index (integer, matching one of the listed criterion indices)\n"
        "- score (integer 1-10)\n"
        "- justification (ONE factual sentence)\n"
        "Do NOT rename options or criteria. Do NOT omit any combinations."
    )

    try:
        structured_llm = llm.with_structured_output(ResearchBatchOutput)
        result: ResearchBatchOutput = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate AI research output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while calling research LLM: {exc}") from exc

    # Additional safety: ensure we have exactly one item per (option, criterion)
    # pair, with no duplicates and all indices in range. If this check fails,
    # the caller (CLI) can fall back to full manual scoring.
    expected_items = len(decision_input.options) * len(decision_input.criteria)
    items = result.items or []

    if len(items) != expected_items:
        raise RuntimeError(
            f"AI returned {len(items)} score items, but expected {expected_items} "
            f"(number_of_options x number_of_criteria)."
        )

    seen_pairs = set()
    for item in items:
        if not (0 <= item.option_index < len(decision_input.options)):
            raise RuntimeError(f"AI returned invalid option_index {item.option_index}.")
        if not (0 <= item.criterion_index < len(decision_input.criteria)):
            raise RuntimeError(f"AI returned invalid criterion_index {item.criterion_index}.")

        pair = (item.option_index, item.criterion_index)
        if pair in seen_pairs:
            raise RuntimeError(
                "AI returned duplicate scores for the same (option_index, criterion_index) pair."
            )
        seen_pairs.add(pair)

    return result


def run_synthesis_llm(
    decision_input: DecisionInputState,
    wsm_result: WSMResult,
) -> SynthesisOutput:
    """
    Call the Groq Llama model to synthesize a short explanation of
    why the winning option scored highest according to the math.
    """
    llm = get_llm()

    system_prompt = (
        "You are an explanation engine. You are given a decision problem, "
        "options, criteria with weights, and the exact numeric results of a "
        "Weighted Sum Model (WSM) calculation. Your job is to write a short, "
        "human-readable explanation (3-6 sentences) of exactly WHY the winner "
        "won, grounded solely in the provided scores and weights.\n"
        "Do NOT introduce new facts or speculation. Only refer to the math, "
        "weights, and observed strengths/weaknesses implied by the numbers."
    )

    options_text = "\n".join(
        f"- {opt.name}: {opt.description or 'No additional description.'}"
        for opt in decision_input.options
    )

    criteria_text = "\n".join(
        f"- {c.name} (weight {c.weight}/10): {c.description or 'No additional description.'}"
        for c in decision_input.criteria
    )

    breakdown_lines: List[str] = []
    for opt in wsm_result.options:
        breakdown_lines.append(f"Option {opt.option_name}: total score {opt.total_score}")
        for contrib in opt.contributions:
            breakdown_lines.append(
                f"  - {contrib.criterion_name}: weight={contrib.weight}, "
                f"score={contrib.score}, contribution={contrib.contribution}"
            )

    breakdown_text = "\n".join(breakdown_lines)

    user_prompt = (
        f"Decision problem:\n{decision_input.problem_description}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Criteria and weights:\n{criteria_text}\n\n"
        f"WSM numeric results (already computed):\n{breakdown_text}\n\n"
        f"Winner: {wsm_result.winner}\n"
        f"Loser (lowest score, if any): {wsm_result.loser or 'N/A'}\n\n"
        "Write a concise explanation in plain language focused on why the winner "
        "won according to the scores and weights. Mention key criteria and how "
        "they influenced the outcome."
    )

    try:
        structured_llm = llm.with_structured_output(SynthesisOutput)
        result: SynthesisOutput = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate synthesis output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while calling synthesis LLM: {exc}") from exc

    return result


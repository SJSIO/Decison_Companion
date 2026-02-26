from __future__ import annotations

import os
from typing import List, Literal, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from .models import DecisionInputState, FuzzyTopsisResult, TriangularFuzzyNumber


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
    score: TriangularFuzzyNumber = Field(
        ...,
        description="Triangular fuzzy score (l, m, u) representing worst-case, most-likely, and best-case values.",
    )
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


class CriterionNature(StrictBaseModel):
    criterion_name: str
    kind: Literal["benefit", "cost"]
    rationale: str = Field(..., min_length=1)


class CriterionNatureBatchOutput(StrictBaseModel):
    items: List[CriterionNature]


def get_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
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


def run_research_llm(
    decision_input: DecisionInputState,
    rag_context: Optional[str] = None,
) -> ResearchBatchOutput:
    """
    Call the Groq Llama model to research each option against each criterion.
    Returns a strictly validated ResearchBatchOutput with fuzzy scores (TFNs).
    If rag_context is provided (from RAG over uploaded PDFs), the model MUST base
    its triangular fuzzy scores (l, m, u) and justifications ONLY on that context.
    """
    llm = get_llm()

    system_prompt = (
        "You are an analytical assistant helping evaluate decision options. "
        "For every combination of option and criterion, you must assign a TRIANGULAR FUZZY score "
        "on a 1–10 scale and provide a concise, factual, one-sentence justification.\n\n"
        "A triangular fuzzy score is represented as three floats (l, m, u) where:\n"
        "- 1.0 <= l <= m <= u <= 10.0\n"
        "- l is the worst-case score, m is the most-likely score, and u is the best-case score.\n"
    )
    if rag_context and rag_context.strip():
        system_prompt += (
            "You have been provided with RELEVANT DOCUMENT EXCERPTS below. You MUST base your "
            "triangular fuzzy scores (l, m, u) and justifications STRICTLY on the information in "
            "those excerpts only. Do NOT use general knowledge or speculation. If the excerpts "
            "do not contain enough information for a given (option, criterion) pair, make a "
            "conservative, clearly justified estimate and widen the fuzzy range to reflect uncertainty.\n\n"
        )
    else:
        system_prompt += (
            "Base your answers only on widely accepted facts and the descriptions provided. "
            "If information is unclear, make a conservative, clearly justified estimate, and widen the "
            "fuzzy range to reflect uncertainty.\n"
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
    )
    if rag_context and rag_context.strip():
        user_prompt += (
            "Relevant document excerpts (base your scores and justifications ONLY on this text):\n"
            "---\n"
            f"{rag_context.strip()}\n"
            "---\n\n"
        )
    user_prompt += (
        "You MUST create one item for EVERY possible combination of option_index "
        "and criterion_index.\n"
        "For each item, you MUST output exactly:\n"
        "- option_index (integer, matching one of the listed option indices)\n"
        "- criterion_index (integer, matching one of the listed criterion indices)\n"
        "- score: a triangular fuzzy score represented as three floats (l, m, u) on the 1–10 scale,\n"
        "         where 1.0 <= l <= m <= u <= 10.0\n"
        "- justification (ONE factual sentence)\n\n"
        "Example item (for illustration only):\n"
        "{\n"
        '  \"option_index\": 0,\n'
        '  \"criterion_index\": 1,\n'
        '  \"score\": {\"l\": 7.0, \"m\": 8.0, \"u\": 9.0},\n'
        '  \"justification\": \"Option 0 scores well on criterion 1 based on reliability and support.\"\n'
        "}\n\n"
        "Do NOT rename options or criteria. Do NOT omit any combinations. Ensure that you "
        "respect the fuzzy score constraints for every item."
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


def classify_criteria_nature(decision_input: DecisionInputState) -> CriterionNatureBatchOutput:
    """
    Use the LLM to classify each criterion as 'benefit' (higher is better)
    or 'cost' (lower is better), with a short rationale.
    """
    llm = get_llm()

    system_prompt = (
        "You classify decision criteria as either 'benefit' (higher values are better) "
        "or 'cost' (lower values are better).\n"
        "Examples:\n"
        "- 'Battery life' -> benefit\n"
        "- 'Price', 'Latency', 'Response time', 'CO2 emissions' -> cost\n"
        "Base your decision only on the criterion names and descriptions provided."
    )

    criteria_block = "\n".join(
        f"- {idx}: {c.name} — {c.description or 'No additional description.'}"
        for idx, c in enumerate(decision_input.criteria)
    )

    user_prompt = (
        "Classify each of the following criteria:\n"
        f"{criteria_block}\n\n"
        "For each criterion, output:\n"
        "- criterion_name: exactly as given\n"
        "- kind: 'benefit' or 'cost'\n"
        "- rationale: ONE sentence justifying your choice"
    )

    try:
        structured_llm = llm.with_structured_output(CriterionNatureBatchOutput)
        result: CriterionNatureBatchOutput = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate criterion nature output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while classifying criteria nature: {exc}") from exc

    return result


def run_synthesis_llm(
    decision_input: DecisionInputState,
    topsis_result: FuzzyTopsisResult,
) -> SynthesisOutput:
    """
    Call the Groq Llama model to synthesize a two-part markdown explanation:
    1) Algorithmic Breakdown of why the winner achieved the highest closeness
       coefficient based on fuzzy scores, weights, and distances.
    2) Contextual Fit explaining how the winner aligns with the user's overall
       goal and the criterion descriptions.
    """
    llm = get_llm()

    system_prompt = (
        "You are an explanation engine. You are given a decision problem, "
        "options, criteria with weights, fuzzy scores (as triangular fuzzy numbers), "
        "and the results of a Fuzzy TOPSIS calculation.\n\n"
        "You MUST produce a markdown explanation with TWO sections, in this order:\n"
        "## Algorithmic Breakdown\n"
        "- Explain exactly why the winner has the highest closeness coefficient.\n"
        "- Refer to distances to the positive/negative ideal solutions and the "
        "criterion weights.\n"
        "- Stay strictly grounded in the math: fuzzy scores, weights, distances, "
        "and closeness coefficients.\n\n"
        "## Contextual Fit\n"
        "- Explain how this mathematically winning option fits the user's overall "
        "goal and the criterion descriptions.\n"
        "- Use only the provided goal/context and criterion descriptions; do NOT "
        "introduce new facts or speculative scenarios.\n"
        "Do NOT reference Fuzzy TOPSIS by name; just explain the reasoning."
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
    for opt_result in topsis_result.options:
        breakdown_lines.append(
            f"Option {opt_result.option_name}: "
            f"distance_to_fpis={opt_result.distance_to_fpis:.4f}, "
            f"distance_to_fnis={opt_result.distance_to_fnis:.4f}, "
            f"closeness_coefficient={opt_result.closeness_coefficient:.4f}"
        )

    breakdown_text = "\n".join(breakdown_lines)

    user_prompt = (
        f"Decision problem (overall goal/context):\n{decision_input.problem_description}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Criteria and weights:\n{criteria_text}\n\n"
        "Fuzzy TOPSIS results (for each option):\n"
        f"{breakdown_text}\n\n"
        f"Winner: {topsis_result.winner}\n"
        f"Loser (lowest score, if any): {topsis_result.loser or 'N/A'}\n\n"
        "Write your answer in exactly TWO markdown sections with these headings:\n"
        "## Algorithmic Breakdown\n"
        "- Focus on distances to the ideal solutions, closeness coefficients, and "
        "how the weights interact with the fuzzy scores.\n\n"
        "## Contextual Fit\n"
        "- Explain why this mathematically winning option fits the user's stated "
        "overall goal and the criterion descriptions.\n"
        "Do not add new information beyond what is implied by the numbers, goal, "
        "and descriptions."
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


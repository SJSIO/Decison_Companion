from __future__ import annotations

import os
from typing import List, Literal, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

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


class ResearchItemLLMOutput(StrictBaseModel):
    """
    Raw LLM output for one (option, criterion) pair: requires chain-of-thought
    extracted_evidence and forbids fallback phrasing. Converted to OptionCriterionAIResearch
    before returning from run_research_llm.
    """
    option_index: int = Field(..., ge=0)
    criterion_index: int = Field(..., ge=0)
    extracted_evidence: str = Field(
        ...,
        min_length=1,
        description="1-2 exact phrases or concepts from the context that relate to the criterion.",
    )
    lower_bound: float = Field(..., ge=1.0, le=10.0, description="Worst-case score.")
    most_likely: float = Field(..., ge=1.0, le=10.0, description="Most realistic score.")
    upper_bound: float = Field(..., ge=1.0, le=10.0, description="Best-case score.")
    justification: str = Field(..., min_length=1)

    @field_validator("extracted_evidence", "justification")
    def strip_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Must not be empty.")
        return v.strip()

    @model_validator(mode="after")
    def bounds_ordered(self) -> "ResearchItemLLMOutput":
        if not (self.lower_bound <= self.most_likely <= self.upper_bound):
            raise ValueError("Require lower_bound <= most_likely <= upper_bound.")
        return self


class ResearchBatchOutputLLM(StrictBaseModel):
    """LLM-structured output with extracted_evidence per item; converted to ResearchBatchOutput."""
    items: List[ResearchItemLLMOutput]


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
    Uses a universal Expert Analyst persona: infers scores from document context and
    broad knowledge; forbids \"not known\" / \"insufficient data\"; injects the user's
    criterion description so scoring matches what the user cares about. Returns a
    strictly validated ResearchBatchOutput with triangular fuzzy scores (l, m, u).
    """
    llm = get_llm()

    system_prompt = (
        "You are an Expert Analyst and Strategic Decision Consultant. Your job is to objectively "
        "evaluate specific options based on provided documentation and your broad industry knowledge. "
        "You work in any domain: vendor contracts, candidates, products, strategies, or any other "
        "decision with options and criteria. For each (option, criterion) pair, use the option name, "
        "the criterion name, and the USER'S DEFINITION of that criterion (provided in the task). "
        "Score strictly according to what the user means by that criterion.\n\n"
        "INSTRUCTIONS AND RULES:\n"
        "1. DEDUCE AND INFER: Documents rarely state perfect answers explicitly. You are STRICTLY "
        "FORBIDDEN from outputting \"not known\", \"not mentioned\", \"insufficient data\", \"lack of "
        "description\", or \"not explicitly mentioned\". You MUST use your expert knowledge to infer "
        "how well this option performs based on the provided context.\n"
        "   - Example: If evaluating a software tool for \"Security\" and the context mentions "
        "\"SOC2 compliance and end-to-end encryption\", infer a high score even if the word "
        "\"Security\" is not explicitly written.\n"
        "   - Example: If the criterion is \"Cost\" and the user's definition is \"Lowest long-term "
        "maintenance fee\", score based on maintenance-related evidence, not just upfront price.\n"
        "2. TRIANGULAR FUZZY SCORING: Output a Triangular Fuzzy Number (l, m, u) on a scale of 1 to 10. "
        "lower_bound (l) = worst-case / most pessimistic; most_likely (m) = most realistic; "
        "upper_bound (u) = best-case / most optimistic. Require l <= m <= u.\n"
        "3. SPARSE CONTEXT: If the context is truly sparse for an option, infer a reasonable baseline "
        "based on the nature of the option and the criterion, and widen your confidence range "
        "(e.g. output a broader triangle like 3, 5, 8). Do not default to the middle; commit to a "
        "plausible range.\n"
        "4. JUSTIFICATION: Provide a 1–2 sentence justification that references the provided text "
        "or your logical deduction. For extracted_evidence, use 1–2 exact phrases from the context "
        "when available; if inferring from thin context, briefly state the inference basis.\n\n"
        "OUTPUT: Return one item per (option_index, criterion_index) with extracted_evidence, "
        "lower_bound, most_likely, upper_bound, and justification. Never refuse to score; always "
        "produce a defensible fuzzy score and justification.\n"
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
        user_def = (crit.description or "").strip() or "Not specified; use the criterion name and context."
        criteria_block_lines.append(
            f"Criterion {idx}: **{crit.name}** (weight {crit.weight}/10). "
            f"**User's description of this criterion:** {user_def}"
        )
    criteria_block = "\n".join(criteria_block_lines)

    user_prompt = (
        f"Decision problem:\n{decision_input.problem_description}\n\n"
        f"Options (indexed by option_index):\n{options_block}\n\n"
        f"Criteria (indexed by criterion_index):\n{criteria_block}\n\n"
    )
    if rag_context and rag_context.strip():
        user_prompt += (
            "Here is the factual context retrieved from the user's uploaded documents:\n"
            "<context>\n"
            f"{rag_context.strip()}\n"
            "</context>\n\n"
            "Score like an objective evaluator: for each criterion, options with clear supporting "
            "evidence in the context must receive higher fuzzy scores (l, m, u); options with no or "
            "weak evidence must receive lower scores. Use the User's Definition of each criterion to "
            "decide what counts as evidence.\n\n"
            "If the context is grouped by 'Context for Option 0', 'Context for Option 1', etc., use "
            "only the section for that option when scoring that option — do not use another option's "
            "context.\n\n"
        )
    user_prompt += (
        "You MUST create one item for EVERY possible combination of option_index and criterion_index. "
        "For each item return:\n"
        "- option_index (integer)\n"
        "- criterion_index (integer)\n"
        "- extracted_evidence: 1–2 exact phrases or concepts from the context that relate to the "
        "criterion; if context is sparse, state the inference basis in a short phrase (e.g. \"Inferred "
        "from SLA and uptime guarantees\")\n"
        "- lower_bound: float 1–10 (worst-case)\n"
        "- most_likely: float 1–10 (most realistic)\n"
        "- upper_bound: float 1–10 (best-case); must have lower_bound <= most_likely <= upper_bound\n"
        "- justification: 1–2 sentences that reference the provided text or your logical deduction. "
        "Do NOT say the information is missing or unknown.\n\n"
        "Example item:\n"
        "{\n"
        '  \"option_index\": 0,\n'
        '  \"criterion_index\": 1,\n'
        '  \"extracted_evidence\": \"SOC2 Type II; encryption at rest and in transit\",\n'
        '  \"lower_bound\": 7.0,\n'
        '  \"most_likely\": 8.5,\n'
        '  \"upper_bound\": 9.0,\n'
        '  \"justification\": \"Compliance and encryption signals strongly support a high security score.\"\n'
        "}\n\n"
        "Do NOT omit any combinations. Do NOT output \"not known\", \"not mentioned\", \"insufficient "
        "data\", or \"lack of description\". Always produce a defensible score and justification.\n"
    )

    try:
        structured_llm = llm.with_structured_output(ResearchBatchOutputLLM)
        raw_result: ResearchBatchOutputLLM = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate AI research output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while calling research LLM: {exc}") from exc

    # Convert LLM output (extracted_evidence, lower_bound, most_likely, upper_bound) to pipeline
    # format (score as TriangularFuzzyNumber, justification). Optionally fold extracted_evidence
    # into justification so the user sees it.
    items: List[OptionCriterionAIResearch] = []
    for raw in raw_result.items or []:
        score_tfn = TriangularFuzzyNumber(
            l=raw.lower_bound,
            m=raw.most_likely,
            u=raw.upper_bound,
        )
        justification = raw.justification
        if raw.extracted_evidence.strip():
            justification = f"[Evidence: {raw.extracted_evidence.strip()}] {justification}"
        items.append(
            OptionCriterionAIResearch(
                option_index=raw.option_index,
                criterion_index=raw.criterion_index,
                score=score_tfn,
                justification=justification.strip(),
            )
        )
    result = ResearchBatchOutput(items=items)

    # Additional safety: ensure we have exactly one item per (option, criterion)
    # pair, with no duplicates and all indices in range.
    expected_items = len(decision_input.options) * len(decision_input.criteria)

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
        "You MUST produce a markdown explanation with TWO sections, in this order:\n\n"
        "## Algorithmic Breakdown\n"
        "Explain step-by-step how the algorithm reached the selected winner. Use numbered steps:\n"
        "1. **Normalization**: How the raw fuzzy scores were normalized per criterion type "
        "(Benefit: higher is better, divide by max; Cost: lower is better, flip and scale). "
        "Mention that values end up in [0, 1].\n"
        "2. **Weighting**: How each criterion's weight (1–10) was applied to the normalized "
        "values to get the weighted matrix.\n"
        "3. **Ideal solutions**: How the Positive Ideal (FPIS) and Negative Ideal (FNIS) "
        "were determined from the weighted matrix (best and worst value per criterion).\n"
        "4. **Distances**: How each option's distance to FPIS and to FNIS was computed "
        "(vertex method: square root of average of squared differences of l, m, u). "
        "Use the actual distance numbers from the results where available.\n"
        "5. **Closeness coefficient (CC)**: How CC = distance_to_FNIS / (distance_to_FPIS + "
        "distance_to_FNIS) for each option; higher CC means closer to the positive ideal.\n"
        "6. **Why this winner**: Explain concretely why the chosen option has the highest CC "
        "and is therefore selected (refer to its distances and CC vs. the others).\n"
        "Stay strictly grounded in the math; use concrete numbers from the provided results.\n\n"
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
        "Use numbered steps (1–6) as in the instructions: normalization, weighting, "
        "FPIS/FNIS, distances, closeness coefficient, and why this winner was selected. "
        "Include concrete numbers from the results above where relevant.\n\n"
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


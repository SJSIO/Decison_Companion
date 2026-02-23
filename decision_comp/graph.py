from __future__ import annotations

from typing import Dict, Tuple

from langgraph.graph import StateGraph, END

from .llm_services import run_research_llm, SynthesisOutput
from .models import (
    AIResearchResultState,
    CriterionContribution,
    DecisionInputState,
    FinalScoresState,
    GraphState,
    OptionCriterionScore,
    OptionWSMResult,
    WSMResult,
)


def ai_research_node(state: GraphState) -> GraphState:
    if state.inputs is None:
        raise ValueError("GraphState.inputs must be set before calling ai_research_node.")

    research_output = run_research_llm(state.inputs)

    scores: Dict[Tuple[str, str], OptionCriterionScore] = {}

    options = state.inputs.options
    criteria = state.inputs.criteria
    num_options = len(options)
    num_criteria = len(criteria)

    for item in research_output.items:
        if not (0 <= item.option_index < num_options):
            raise ValueError(f"AI returned invalid option_index {item.option_index}.")
        if not (0 <= item.criterion_index < num_criteria):
            raise ValueError(f"AI returned invalid criterion_index {item.criterion_index}.")

        opt_name = options[item.option_index].name
        crit_name = criteria[item.criterion_index].name
        key = (opt_name, crit_name)
        scores[key] = OptionCriterionScore(
            option_name=opt_name,
            criterion_name=crit_name,
            score=item.score,
            justification=item.justification,
        )

    state.ai_scores = AIResearchResultState(scores=scores)
    return state


def compute_wsm(decision_input: DecisionInputState, final_scores: FinalScoresState) -> WSMResult:
    """
    Deterministic Weighted Sum Model (WSM) implementation.
    No randomness, no external calls, purely numeric.
    """
    # Build fast lookup for criterion weights.
    weight_by_criterion: Dict[str, int] = {c.name: c.weight for c in decision_input.criteria}

    # Ensure all required scores are present.
    for opt in decision_input.options:
        for crit in decision_input.criteria:
            key = (opt.name, crit.name)
            if key not in final_scores.scores:
                raise ValueError(f"Missing score for option '{opt.name}' and criterion '{crit.name}'.")

    option_results: Dict[str, OptionWSMResult] = {}

    for opt in decision_input.options:
        contributions = []
        total = 0
        for crit in decision_input.criteria:
            key = (opt.name, crit.name)
            score_entry = final_scores.scores[key]
            weight = weight_by_criterion[crit.name]
            contribution = weight * score_entry.score
            total += contribution
            contributions.append(
                CriterionContribution(
                    criterion_name=crit.name,
                    weight=weight,
                    score=score_entry.score,
                    contribution=contribution,
                )
            )
        option_results[opt.name] = OptionWSMResult(
            option_name=opt.name,
            total_score=total,
            contributions=contributions,
        )

    # Rank options by total_score (desc), then by option_name (asc) for deterministic ordering.
    sorted_options = sorted(
        option_results.values(),
        key=lambda o: (-o.total_score, o.option_name),
    )

    winner = sorted_options[0].option_name
    loser = sorted_options[-1].option_name if len(sorted_options) > 1 else None

    return WSMResult(options=sorted_options, winner=winner, loser=loser)


def wsm_calculation_node(state: GraphState) -> GraphState:
    if state.inputs is None or state.final_scores is None:
        raise ValueError(
            "GraphState.inputs and GraphState.final_scores must be set before calling wsm_calculation_node."
        )

    state.wsm_result = compute_wsm(state.inputs, state.final_scores)
    return state


def synthesis_node(state: GraphState) -> GraphState:
    from .llm_services import run_synthesis_llm

    if state.inputs is None or state.wsm_result is None:
        raise ValueError("GraphState.inputs and GraphState.wsm_result must be set before calling synthesis_node.")

    synthesis: SynthesisOutput = run_synthesis_llm(state.inputs, state.wsm_result)
    state.explanation = synthesis.explanation
    return state


def build_decision_graph() -> StateGraph:
    """
    Build and return a LangGraph StateGraph for the automated parts of the workflow.
    Human-in-the-loop editing of scores happens outside of this graph in the CLI.
    """
    graph = StateGraph(GraphState)

    graph.add_node("ai_research", ai_research_node)
    graph.add_node("wsm_calculation", wsm_calculation_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("ai_research")
    graph.add_edge("ai_research", "wsm_calculation")
    graph.add_edge("wsm_calculation", "synthesis")
    graph.add_edge("synthesis", END)

    return graph


def run_ai_research(inputs: DecisionInputState) -> AIResearchResultState:
    """
    Convenience function: run only the AI research node.
    Intended to be called from the CLI before the human-in-the-loop step.
    """
    state = GraphState(inputs=inputs)
    updated_state = ai_research_node(state)
    if updated_state.ai_scores is None:
        raise RuntimeError("AI research did not produce any scores.")
    return updated_state.ai_scores


def run_calculation_and_synthesis(
    inputs: DecisionInputState,
    final_scores: FinalScoresState,
) -> Tuple[WSMResult, str]:
    """
    Convenience function: run deterministic WSM calculation and synthesis explanation.
    """
    state = GraphState(inputs=inputs, final_scores=final_scores)
    state = wsm_calculation_node(state)
    state = synthesis_node(state)
    if state.wsm_result is None or state.explanation is None:
        raise RuntimeError("Calculation and synthesis did not complete successfully.")
    return state.wsm_result, state.explanation


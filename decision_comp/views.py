"""
Django API endpoints for the Decision Companion (headless backend).
Streamlit (or other clients) call these to run research and calculation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .graph import run_ai_research, run_calculation_and_synthesis
from .models import (
    CriterionSchema,
    DecisionInputState,
    FinalScoresState,
    OptionCriterionScore,
    OptionSchema,
    TriangularFuzzyNumber,
)


def _parse_json_body(request) -> dict:
    """Parse JSON from request body; return 400 on error."""
    try:
        return json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None


def _research_payload_to_inputs(data: dict) -> DecisionInputState:
    """Build DecisionInputState from API payload for research."""
    problem = data.get("problem_description") or ""
    options_data = data.get("options") or []
    criteria_data = data.get("criteria") or []

    if len(options_data) < 2:
        raise ValueError("At least 2 options are required.")
    if len(criteria_data) < 1:
        raise ValueError("At least 1 criterion is required.")

    options = [
        OptionSchema(name=o["name"], description=o.get("description"))
        for o in options_data
    ]
    criteria = []
    for c in criteria_data:
        weight = int(c.get("weight", 5))
        if not (1 <= weight <= 10):
            raise ValueError(f"Criterion weight must be 1–10, got {weight}.")
        kind = (c.get("kind") or "benefit").lower()
        if kind not in ("benefit", "cost"):
            kind = "benefit"
        criteria.append(
            CriterionSchema(
                name=c["name"],
                weight=weight,
                description=c.get("description"),
                kind=kind,
            )
        )

    return DecisionInputState(
        problem_description=problem,
        options=options,
        criteria=criteria,
    )


def _scores_list_to_final_scores(
    inputs: DecisionInputState,
    scores_list: List[Dict[str, Any]],
) -> FinalScoresState:
    """Build FinalScoresState from a list of score objects (e.g. from Streamlit grid)."""
    scores: Dict[Tuple[str, str], OptionCriterionScore] = {}
    for row in scores_list:
        opt_name = row["option_name"]
        crit_name = row["criterion_name"]
        l_val = float(row["l"])
        m_val = float(row["m"])
        u_val = float(row["u"])
        justification = (row.get("justification") or "").strip() or "No justification provided."
        tfn = TriangularFuzzyNumber(l=l_val, m=m_val, u=u_val)
        key = (opt_name, crit_name)
        scores[key] = OptionCriterionScore(
            option_name=opt_name,
            criterion_name=crit_name,
            score_tfn=tfn,
            justification=justification,
        )

    return FinalScoresState(scores=scores)


@csrf_exempt
@require_http_methods(["POST"])
def api_research(request):
    """
    POST /api/research/
    Body: { problem_description, options: [{name, description?}], criteria: [{name, weight, description?, kind?}] }
    Returns: { scores: [{ option_name, criterion_name, l, m, u, justification }] }
    """
    data = _parse_json_body(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    try:
        inputs = _research_payload_to_inputs(data)
    except (KeyError, ValueError, TypeError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    try:
        ai_result = run_ai_research(inputs)
    except Exception as e:
        return JsonResponse({"error": f"AI research failed: {e}"}, status=500)

    # Serialize scores to a list (JSON does not support tuple keys).
    scores_list = []
    for (opt_name, crit_name), score_obj in ai_result.scores.items():
        scores_list.append({
            "option_name": opt_name,
            "criterion_name": crit_name,
            "l": score_obj.score_tfn.l,
            "m": score_obj.score_tfn.m,
            "u": score_obj.score_tfn.u,
            "justification": score_obj.justification,
        })

    return JsonResponse({"scores": scores_list})


@csrf_exempt
@require_http_methods(["POST"])
def api_calculate(request):
    """
    POST /api/calculate/
    Body: { problem_description, options: [...], criteria: [...], scores: [{ option_name, criterion_name, l, m, u, justification }] }
    Returns: { winner, loser, explanation, options: [{ option_name, closeness_coefficient, distance_to_fpis, distance_to_fnis }] }
    """
    data = _parse_json_body(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    try:
        inputs = _research_payload_to_inputs(data)
    except (KeyError, ValueError, TypeError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    scores_list = data.get("scores")
    if not scores_list:
        return JsonResponse({"error": "Missing or empty 'scores' array."}, status=400)

    try:
        final_scores = _scores_list_to_final_scores(inputs, scores_list)
    except (KeyError, ValueError, TypeError) as e:
        return JsonResponse({"error": f"Invalid score data: {e}"}, status=400)

    try:
        topsis_result, explanation = run_calculation_and_synthesis(inputs, final_scores)
    except Exception as e:
        return JsonResponse({"error": f"Calculation and synthesis failed: {e}"}, status=500)

    options_out = [
        {
            "option_name": o.option_name,
            "closeness_coefficient": o.closeness_coefficient,
            "distance_to_fpis": o.distance_to_fpis,
            "distance_to_fnis": o.distance_to_fnis,
        }
        for o in topsis_result.options
    ]

    return JsonResponse({
        "winner": topsis_result.winner,
        "loser": topsis_result.loser,
        "explanation": explanation,
        "options": options_out,
    })

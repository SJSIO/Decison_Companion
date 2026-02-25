"""
Decision Companion — Streamlit frontend.
Run: streamlit run streamlit_app.py
Ensure Django API is running on API_BASE_URL (default http://localhost:8000).
"""
from __future__ import annotations

import requests
import streamlit as st
import pandas as pd

API_BASE_URL = "http://localhost:8000"


def _ensure_session_state():
    if "options" not in st.session_state:
        st.session_state.options = []
    if "criteria" not in st.session_state:
        st.session_state.criteria = []
    if "problem_description" not in st.session_state:
        st.session_state.problem_description = ""
    if "research_scores" not in st.session_state:
        st.session_state.research_scores = None
    if "edited_df" not in st.session_state:
        st.session_state.edited_df = None
    if "calculate_result" not in st.session_state:
        st.session_state.calculate_result = None


def _api_research(problem_description: str, options: list, criteria: list) -> dict:
    payload = {
        "problem_description": problem_description,
        "options": [{"name": o["name"], "description": o.get("description") or ""} for o in options],
        "criteria": [
            {"name": c["name"], "weight": c["weight"], "description": c.get("description") or "", "kind": c.get("kind") or "benefit"}
            for c in criteria
        ],
    }
    r = requests.post(f"{API_BASE_URL}/api/research/", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _api_calculate(problem_description: str, options: list, criteria: list, scores: list) -> dict:
    payload = {
        "problem_description": problem_description,
        "options": [{"name": o["name"], "description": o.get("description") or ""} for o in options],
        "criteria": [
            {"name": c["name"], "weight": c["weight"], "description": c.get("description") or "", "kind": c.get("kind") or "benefit"}
            for c in criteria
        ],
        "scores": scores,
    }
    r = requests.post(f"{API_BASE_URL}/api/calculate/", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    _ensure_session_state()

    st.set_page_config(page_title="Decision Companion", layout="wide")
    st.title("Decision Companion")
    st.caption("Define options and criteria, run AI research, edit fuzzy scores, then run the final decision.")

    # ----- Phase 1: Inputs -----
    st.header("Phase 1: Options and criteria")

    problem = st.text_area(
        "Problem description (optional)",
        value=st.session_state.problem_description,
        key="problem_input",
        height=80,
    )
    st.session_state.problem_description = problem

    col_crit, col_opt = st.columns(2)

    with col_crit:
        st.subheader("Criteria")
        new_crit_name = st.text_input("Criterion name", key="new_criterion_name")
        new_crit_weight = st.slider("Weight (1–10)", 1, 10, 5, key="new_criterion_weight")
        if st.button("Add criterion") and new_crit_name.strip():
            st.session_state.criteria.append({"name": new_crit_name.strip(), "weight": new_crit_weight})
            st.rerun()
        for i, c in enumerate(st.session_state.criteria):
            st.text(f"• {c['name']} (weight {c['weight']})")
            if st.button("Remove", key=f"rem_crit_{i}"):
                st.session_state.criteria.pop(i)
                st.rerun()

    with col_opt:
        st.subheader("Options")
        new_opt_name = st.text_input("Option name", key="new_option_name")
        if st.button("Add option") and new_opt_name.strip():
            st.session_state.options.append({"name": new_opt_name.strip()})
            st.rerun()
        for i, o in enumerate(st.session_state.options):
            st.text(f"• {o['name']}")
            if st.button("Remove", key=f"rem_opt_{i}"):
                st.session_state.options.pop(i)
                st.rerun()

    if len(st.session_state.options) < 2 or len(st.session_state.criteria) < 1:
        st.info("Add at least 2 options and 1 criterion to continue.")
        return

    # ----- Phase 2: Research -----
    st.header("Phase 2: AI research")
    if st.button("Run AI research"):
        with st.spinner("AI agents researching and validating data..."):
            try:
                result = _api_research(
                    st.session_state.problem_description,
                    st.session_state.options,
                    st.session_state.criteria,
                )
                st.session_state.research_scores = result.get("scores") or []
                st.session_state.edited_df = None
                st.success("Research complete. Review and edit scores below.")
            except requests.RequestException as e:
                st.error(f"Research request failed: {e}")
            except Exception as e:
                st.error(str(e))
        st.rerun()

    # ----- Phase 3: Human-in-the-loop -----
    st.header("Phase 3: Review and edit scores")
    scores = st.session_state.research_scores
    if scores:
        df = pd.DataFrame(scores)
        if st.session_state.edited_df is not None:
            df = st.session_state.edited_df
        edited = st.data_editor(
            df,
            use_container_width=True,
            key="scores_editor",
            column_config={
                "option_name": st.column_config.TextColumn("Option", disabled=True),
                "criterion_name": st.column_config.TextColumn("Criterion", disabled=True),
                "l": st.column_config.NumberColumn("l", min_value=1.0, max_value=10.0, format="%.2f"),
                "m": st.column_config.NumberColumn("m", min_value=1.0, max_value=10.0, format="%.2f"),
                "u": st.column_config.NumberColumn("u", min_value=1.0, max_value=10.0, format="%.2f"),
                "justification": st.column_config.TextColumn("Justification"),
            },
        )
        st.session_state.edited_df = edited

        # ----- Phase 4: Calculation -----
        st.header("Phase 4: Final decision")
        if st.button("Run final decision"):
            # Use edited dataframe as scores list.
            scores_payload = edited.to_dict(orient="records")
            with st.spinner("Computing Fuzzy TOPSIS and synthesis..."):
                try:
                    calc = _api_calculate(
                        st.session_state.problem_description,
                        st.session_state.options,
                        st.session_state.criteria,
                        scores_payload,
                    )
                    st.session_state.calculate_result = calc
                    st.success("Calculation complete.")
                except requests.RequestException as e:
                    st.error(f"Calculation request failed: {e}")
                except Exception as e:
                    st.error(str(e))
            st.rerun()
    else:
        st.info("Run AI research first to see the scores grid.")

    # ----- Phase 5: Results -----
    result = st.session_state.calculate_result
    if result:
        st.header("Phase 5: Results")
        winner = result.get("winner")
        explanation = result.get("explanation", "")
        options_cc = result.get("options") or []

        if winner:
            st.success(f"Winner: **{winner}**")
            st.metric("Recommended option", winner)

        if explanation:
            st.subheader("Explanation")
            st.write(explanation)

        if options_cc:
            st.subheader("Closeness coefficients")
            cc_df = pd.DataFrame(options_cc)
            cc_df = cc_df.sort_values("closeness_coefficient", ascending=False).reset_index(drop=True)
            st.bar_chart(cc_df.set_index("option_name")["closeness_coefficient"])


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict, Tuple

from django.core.management.base import BaseCommand, CommandError

from ...graph import run_ai_research, run_calculation_and_synthesis
from ...llm_services import classify_criteria_nature
from ...models import (
    DecisionInputState,
    FinalScoresState,
    OptionCriterionScore,
    OptionSchema,
    CriterionSchema,
    TriangularFuzzyNumber,
)


def _prompt_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.")


def _prompt_int(prompt: str, min_value: int | None = None, max_value: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be at most {max_value}.")
            continue
        return value


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


class Command(BaseCommand):
    help = "Interactive Decision Companion CLI using a Fuzzy TOPSIS engine and optional AI assistance."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-ai",
            action="store_true",
            help="Skip AI research and enter all scores manually.",
        )

    def handle(self, *args, **options):
        use_ai = not options.get("no_ai", False)

        self.stdout.write(self.style.MIGRATE_HEADING("Decision Companion"))
        self.stdout.write("")

        try:
            decision_input = self._collect_decision_input()
        except Exception as exc:  # pragma: no cover - defensive
            raise CommandError(f"Failed to collect decision input: {exc}") from exc

        ai_scores = None
        if use_ai:
            self.stdout.write("")
            self.stdout.write(self.style.HTTP_INFO("Running AI research to propose scores..."))
            try:
                ai_scores = run_ai_research(decision_input)
            except Exception as exc:
                self.stdout.write(
                    self.style.WARNING(
                        f"AI research failed or returned inconsistent results: {exc}"
                    )
                )
                self.stdout.write(
                    self.style.WARNING(
                        "Continuing with manual scoring for all options and criteria."
                    )
                )
                ai_scores = None
            else:
                self._display_ai_scores(decision_input, ai_scores.scores)
        else:
            self.stdout.write(self.style.WARNING("Skipping AI research phase; you will enter all scores manually."))

        final_scores = self._collect_final_scores(decision_input, ai_scores.scores if ai_scores else None)

        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO("Calculating Fuzzy TOPSIS results and explanation..."))
        try:
            topsis_result, explanation = run_calculation_and_synthesis(decision_input, final_scores)
        except Exception as exc:
            raise CommandError(f"Calculation and synthesis failed: {exc}") from exc

        self._display_topsis_results(topsis_result, explanation)

    def _collect_decision_input(self) -> DecisionInputState:
        problem_description = _prompt_non_empty("Describe the decision problem:\n> ")

        num_options = _prompt_int("How many options do you want to compare? (>=2): ", min_value=2)
        options: list[OptionSchema] = []
        for i in range(num_options):
            self.stdout.write(f"\nOption {i + 1}/{num_options}")
            name = _prompt_non_empty("  Name: ")
            description = input("  Description (optional): ").strip() or None
            options.append(OptionSchema(name=name, description=description))

        num_criteria = _prompt_int("How many criteria do you want to use? (>=1): ", min_value=1)
        criteria: list[CriterionSchema] = []
        for i in range(num_criteria):
            self.stdout.write(f"\nCriterion {i + 1}/{num_criteria}")
            name = _prompt_non_empty("  Name: ")
            description = input("  Description (optional): ").strip() or None
            weight = _prompt_int("  Weight (1-10, higher means more important): ", min_value=1, max_value=10)
            # Temporarily set kind to default; we'll refine via AI + human-in-the-loop.
            criteria.append(CriterionSchema(name=name, weight=weight, description=description))

        # Build preliminary decision input for AI-based criterion classification.
        try:
            decision_input = DecisionInputState(
                problem_description=problem_description,
                options=options,
                criteria=criteria,
            )
        except Exception as exc:
            raise CommandError(f"Invalid decision input: {exc}") from exc

        # Let the AI suggest benefit/cost nature for each criterion, then confirm with the user.
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO("Classifying criteria as benefit or cost..."))
        nature_by_name: Dict[str, str] = {}
        try:
            nature_batch = classify_criteria_nature(decision_input)
            for item in nature_batch.items:
                nature_by_name[item.criterion_name] = item.kind
        except Exception as exc:
            self.stdout.write(
                self.style.WARNING(
                    f"AI criterion classification failed: {exc}. Defaulting all criteria to 'benefit'."
                )
            )

        for crit in criteria:
            suggested_kind = nature_by_name.get(crit.name, "benefit")
            self.stdout.write(f"\nCriterion: {crit.name}")
            self.stdout.write(f"  AI suggests type: {suggested_kind}")
            accept_kind = _prompt_yes_no("  Accept this type?", default_yes=True)
            if accept_kind:
                crit.kind = suggested_kind
            else:
                while True:
                    manual = input("  Enter criterion type ('benefit' or 'cost'): ").strip().lower()
                    if manual in ("benefit", "cost"):
                        crit.kind = manual
                        break
                    self.stdout.write("  Please enter 'benefit' or 'cost'.")

        # Rebuild the decision input with confirmed kinds.
        try:
            return DecisionInputState(
                problem_description=problem_description,
                options=options,
                criteria=criteria,
            )
        except Exception as exc:
            raise CommandError(f"Invalid decision input after criterion classification: {exc}") from exc

    def _display_ai_scores(
        self,
        decision_input: DecisionInputState,
        scores: Dict[Tuple[str, str], OptionCriterionScore],
    ) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_LABEL("AI suggested scores:"))
        for opt in decision_input.options:
            self.stdout.write(f"\nOption: {opt.name}")
            for crit in decision_input.criteria:
                key = (opt.name, crit.name)
                score_entry = scores.get(key)
                if not score_entry:
                    self.stdout.write(f"  [NO SCORE RETURNED] Criterion: {crit.name}")
                    continue
                tfn = score_entry.score_tfn
                self.stdout.write(
                    f"  Criterion: {crit.name} | "
                    f"Fuzzy score: [{tfn.l:.2f}, {tfn.m:.2f}, {tfn.u:.2f}] / 10 | "
                    f"Reason: {score_entry.justification}"
                )

    def _collect_final_scores(
        self,
        decision_input: DecisionInputState,
        ai_scores: Dict[Tuple[str, str], OptionCriterionScore] | None,
    ) -> FinalScoresState:
        final_scores = FinalScoresState()
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Human review of scores"))

        for opt in decision_input.options:
            self.stdout.write(f"\nOption: {opt.name}")
            for crit in decision_input.criteria:
                key = (opt.name, crit.name)
                ai_entry = ai_scores.get(key) if ai_scores else None

                if ai_entry:
                    self.stdout.write(
                        f"  Criterion: {crit.name}\n"
                        f"    AI suggested fuzzy score: "
                        f"[{ai_entry.score_tfn.l:.2f}, {ai_entry.score_tfn.m:.2f}, {ai_entry.score_tfn.u:.2f}] / 10\n"
                        f"    Justification: {ai_entry.justification}"
                    )
                    accept = _prompt_yes_no("    Accept this score?", default_yes=True)
                    if accept:
                        final_scores.scores[key] = ai_entry
                        continue
                    else:
                        self.stdout.write("    Enter your own score.")
                else:
                    self.stdout.write(f"  Criterion: {crit.name} (no AI suggestion available)")

                # Manual entry path for fuzzy score.
                while True:
                    raw = input(
                        "    Enter fuzzy score as 'l m u' (floats in 1-10, "
                        "with l <= m <= u), or a single integer 1-10 for a crisp score: "
                    ).strip()
                    parts = raw.split()
                    try:
                        if len(parts) == 1:
                            # Treat as crisp score and expand to a narrow TFN.
                            x = float(parts[0])
                            if not (1.0 <= x <= 10.0):
                                raise ValueError
                            l_val = max(1.0, x - 0.5)
                            u_val = min(10.0, x + 0.5)
                            tfn = TriangularFuzzyNumber(l=l_val, m=x, u=u_val)
                        elif len(parts) == 3:
                            l_val, m_val, u_val = map(float, parts)
                            tfn = TriangularFuzzyNumber(l=l_val, m=m_val, u=u_val)
                        else:
                            raise ValueError
                    except Exception:
                        self.stdout.write(
                            self.style.WARNING(
                                "    Invalid fuzzy score. Please enter either one integer 1-10 "
                                "or three floats l m u with 1.0 <= l <= m <= u <= 10.0."
                            )
                        )
                        continue

                    break

                justification = _prompt_non_empty("    Your one-sentence justification: ")
                final_scores.scores[key] = OptionCriterionScore(
                    option_name=opt.name,
                    criterion_name=crit.name,
                    score_tfn=tfn,
                    justification=justification,
                )

        return final_scores

    def _display_topsis_results(self, topsis_result, explanation: str) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Fuzzy TOPSIS Results"))

        for opt in topsis_result.options:
            self.stdout.write(f"\nOption: {opt.option_name}")
            self.stdout.write(
                f"  Distance to FPIS (D+): {opt.distance_to_fpis:.4f}\n"
                f"  Distance to FNIS (D-): {opt.distance_to_fnis:.4f}\n"
                f"  Closeness coefficient (CC): {opt.closeness_coefficient:.4f}"
            )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Winner: {topsis_result.winner}"))
        if topsis_result.loser:
            self.stdout.write(self.style.WARNING(f"Lowest-scoring option: {topsis_result.loser}"))

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Explanation"))
        self.stdout.write(explanation)


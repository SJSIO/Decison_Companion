How I Started: Defining the Core Engine
My initial goal was to build a system that satisfied the core constraints: accept multiple options, process weighted criteria, and provide a ranked recommendation.

I started by designing a pure mathematical backend using the Weighted Sum Model (WSM). Before introducing any AI, I built a basic Python script where a user manually inputted scores (1-10) for various options. This ensured that the foundation of the system was 100% deterministic and explainable, completely avoiding the "black box" problem where an AI just magically picks a winner.
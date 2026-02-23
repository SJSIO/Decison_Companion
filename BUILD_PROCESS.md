**How I Started: Defining the Core Engine**

23/02/26

Initially I thought of building a domain specific decision companion system which helps a person with multiple offers in different companies, decide which offer to chose based on different criteria's then after the session held to clear doubts on the home assignment, I thought of building a general decision companion system where the user can ask for help in making decisions in any topics concerned and I also made sure that the, in the design, system followed  the core constraints like accept multiple options, process weighted criteria, and provide a ranked recommendation.



Through the researches I done on the topic, I planned to implement a pure mathematical backend using the Weighted Sum Model (WSM). Before introducing any AI, I built a basic Python script where a user manually inputted scores (1-10) for various options.



How it works:

1.Define Criteria \& Weights: The user defines criteria (e.g., Cost, Performance, Learning Curve) and assigns a weight to each (e.g., 1 to 5, or percentages summing to 100%).

2.Score Options: The user scores each option (e.g., React, Vue, Angular) against each criterion on a standard scale (e.g., 1 to 10).

3.Calculate: For each option, multiply the score of each criterion by its weight, and sum them up.



The option with highest value is suggested by the program to the user



Then I asked gemini LLM to simulate a CLI based on the idea provided:



"$ python decision\_maker.py



Welcome to the Decision Companion System!

What are you trying to decide today?

> Choosing a database for my next project



Great. Let's add your criteria.

Enter a criterion name (or type 'done'):

> Performance

Enter a weight for 'Performance' (1-10):

> 9



Enter a criterion name (or type 'done'):

> Ease of Use

Enter a weight for 'Ease of Use' (1-10):

> 6



... \[User enters options like MySQL, ChromaDB, etc. and scores them] ...



--- FINAL RESULTS ---

1\. ChromaDB (Score: 85)

2\. MySQL (Score: 72)



Recommendation: You should choose ChromaDB because its high performance score strongly aligned with your heavily weighted criteria."





##### **How My Thinking Evolved: Solving the "User Burden" Problem**

##### 



###### 23/02/2026

Once the mathematical engine model was fixed, I realized a significant flaw, that the user itself is scoring the different options and  If a user wants to compare options, forcing them to manually research and input the exact scores defeats the purpose of a "Decision Companion.". So I thought by integrating an Ai llm model would be a good idea.



Alternative Approach Considered: I initially considered letting an LLM calculate the final decision entirely by providing inputs to the llm along with passing a prompt asking the AI to "choose the best option based on these weights."

Why I Rejected It: I quickly realized LLMs are notoriously unreliable when we use the same system prompt for differnt scenarios with only the inputs changed as I have experience with llms in building my final year project and also this approach would voilate the assignment's requirement that the logic must be explainable and not rely entirely on an AI model.



The Pivot (The Hybrid Architecture):

I pivoted to a multi-agent architectural approach. I decided to use the LLM strictly as a Research Agent (to gather data and estimate initial scores) and a Synthesis Agent (to explain the final math). The core calculation would remain pure mathematical model implemented in Python code.



###### 24/02/2026

Implemented the project using weight sum model as the core algoritham and used python Django with simple command line interface using Cursor Ai.



I initially went with the Weighted Sum Model (WSM) for the core engine because it’s completely transparent. You multiply the score by the weight, add it up, and you're done. It easily satisfied the "explainable logic" requirement.



But while I was testing the project with different domains the results I got didn't satisfy me.Throuogh various reseach I realized it has a massive blind spot. WSM just averages things out. That means a terrible score in one critical area can get completely hidden if the other scores are high enough.



When I tested evaluating laptops for someone who travels constantly.



The Weights: Performance (9), Display (8), Build Quality (8), Battery Life (8).



I fed two hypothetical laptops into the math engine:



The one laptop which  Scored a solid 8 across the board.



Total WSM Score: 264.



And a gaming laptop Scored a perfect 10 in Performance, Display, and Build, but got a 1 in Battery.



Total WSM Score: 258.



The first laptop won, but it was way too close. I realized that if the user just increased the "Performance" weight from a 9 up to a 10, the Gaming laptop suddenly jumps to 268 and wins the entire evaluation.



That is a real-world system failure. Recommending a laptop with a 30-minute battery life to a traveler is useless. A battery score of 1 is very bad deal , and the WSM couldn't understand that.

So I thought of changing the core algoritham and started researching for it


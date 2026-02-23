##### **GEMINI PROMPTS**





1. I have project assignment to be completed with Problem Statement:



Design and build a “Decision Companion System” that helps a user make better decisions.



The system should assist a user in evaluating options for a real-world decision of their choice.



Your system must:



Accept multiple options



Accept criteria (which may have different weights or importance)



Process and evaluate options against criteria



Provide a ranked recommendation



Explain why a particular recommendation was made

Give me a very detailed idea on how the core design and algoritham to be used in the project should be and the various approaches I can consider on building this project.



**2.** 

Explain clearly how the weigh sum model algoritham can be used to implement this project making sure the constraints like the system must accept multiple options, accepts criteria, process and evaluate options against criteria is met.





**3**.

Do a simulation of working of this project in a command line interface.



**4.** 

In the simulation provided the score value and the weigh value is provided by the user, shouldn't the score be determined the system itself?



**5.** 

I chose to build a universal decision engine which is ai powered in order to get score values on various options based on ai research. so now  provide me with core design and  architecture on how to implement this project by keeping this conditions not violated



System must:



Accept multiple options

Accept criteria (which may have different weights or importance)

Process and evaluate options against criteria

Provide a ranked recommendation

Explain why a particular recommendation was made

The system should not be a static hard coded comparison.

The user should be able to change inputs and get different outcomes.

Your logic should be explainable (not a black box).



The roles of Ai in this project include:



  1.To research on the various options provided by the user and score them, or search for various option along with the scores when the user has not provided options

  2.The calculated result from the weigh sum model  is passed on to ai for final refinement of output generation with a ranked recommendation  that clearly states the reason why a particular decision was made based on the algorithms implemented and research done by ai models



**6.**

now i need you to generate a prompt to build this project with just a Command line interface as frontend where the backend is build using with python django and langchain and langraph framework where the llm model used is the llama model from groq cloud api and use the weigh sum model as the core algorithm

##### **CURSOR AI PROMPTS**

#####  

######   **1.**

Act as an expert Staff Software Engineer and System Architect. I need to build a "Decision Companion System" as a take-home assignment. 



The system must evaluate multiple options against weighted criteria, rank them, and explain the decision. It must not act as a black box; the core decision logic must be deterministic math, while AI is used strictly for data gathering and synthesis.



TECH STACK:

\- Framework: Python / Django

\- Frontend: CLI (implemented via a Django Custom Management Command)

\- AI Orchestration: LangChain \& LangGraph

\- LLM: Llama-3 (via Groq Cloud API)

\- Data Contracts: Pydantic



CORE ALGORITHM:

Weighted Sum Model (WSM). For each option, multiply the criterion score (1-10) by the criterion weight, and sum the results to get the total score.



ARCHITECTURE \& DATA FLOW (LangGraph State):

1\. User Input Phase (CLI): The user defines the problem, options, and criteria (with 1-10 weights) via terminal inputs.

2\. AI Research Node: A LangGraph node takes the inputs and calls the Groq Llama model using `with\_structured\_output` and a strict Pydantic schema. The AI researches each option against each criterion, returning a strict integer score (1-10) and a 1-sentence factual justification.

3\. Human-in-the-Loop Phase (CLI): The CLI displays the AI's suggested scores. The user MUST be prompted to either accept the AI's scores or manually overwrite them. This ensures the system is not hard-coded and the user retains control.

4\. Calculation Node: A pure Python deterministic function applies the WSM algorithm to the finalized scores and ranks the options.

5\. Synthesis Node: A LangGraph node takes the WSM math results (winner, loser, score breakdown) and calls the Groq Llama model to write a short, human-readable summary explaining exactly \*why\* the winner won based on the math and weights.



DELIVERABLES REQUIRED:

Please write the complete code for this project, including:

1\. `models.py`: Django models (if necessary for state) or Pydantic state schemas for LangGraph.

2\. `graph.py`: The complete LangGraph workflow defining the nodes, edges, and state.

3\. `llm\_services.py`: The LangChain setup using `ChatGroq` and the Pydantic schemas for structured output.

4\. `management/commands/decide.py`: The Django custom management command that acts as the interactive CLI, handles the `input()` loops, triggers the LangGraph workflow, and handles the Human-in-the-Loop editing step.



Focus on clean, readable, modular code. Include robust error handling for the CLI inputs and the JSON parsing from the Groq API.



###### **2.**



I'm working on a file called decision\_comp/llm\_services.py for my project. I'm using this library called Pydantic to check my inputs, but my code keeps crashing before the app even starts.



###### **3.**



Explain clearly how to set up the virtual environment in python and to run this project step by step ensuring that the project runs without any erroe, and the contents of requirement.txt which is needed to be downloaded and also where to set up the groq api key in the project folder


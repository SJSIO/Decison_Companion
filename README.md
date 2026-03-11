# Decision Companion

A multi-agent decision-making system combining AI research with the Fuzzy TOPSIS algorithm for ranked, explainable recommendations. Users define options and weighted criteria. Crucially, if the LLM lacks the pre-trained knowledge to evaluate the criteria, users can upload custom PDFs to trigger local decision-making—forcing the AI to score, rank, and explain the best choice based strictly on your provided documents.

## Architecture

```
Streamlit (Frontend)  ──HTTP──>  Django (Backend API)
       │                              │
       │                         ┌────┴─────┐
       │                    LangGraph       ChromaDB
       │                   (Agents)      (Vector Store)
       │                   ┌───┴───┐
       │              Research   Synthesis
       │              Agent      Agent
       │                │
       │           Groq / Llama 3
       │
  Fuzzy TOPSIS
  (Deterministic Math)
```

At first glance, using Django and LangGraph might look like overkill, but it solves a lot of real-world problems.I used django, because it can designed as an api and very easily integrated to any frontend. For the AI side, LangGraph is essential because getting accurate numbers for the Fuzzy TOPSIS math requires more than a simple linear prompt and also to implement RAG it was necessary. Building it as a graph handles the orchestration right now, and it lays the groundwork for a future feature: adding a 'Critic Agent' that can dynamically loop back and fix LLM hallucinations on the fly.


**Frontend** — Streamlit multipage app with a five-phase workflow.
**Backend** — Django REST API exposing `/api/research/` and `/api/calculate/`.
**AI Agents** — LangGraph orchestrates a Research Agent (scores options against criteria using LLM + RAG context) and a Synthesis Agent (explains the mathematical result in plain language).
**Core Algorithm** — Fuzzy TOPSIS using triangular fuzzy numbers, implemented in pure Python with no AI dependency.
**RAG Pipeline** — Per-option PDF uploads are chunked, embedded locally with a HuggingFace model, stored in an ephemeral ChromaDB collection, and retrieved per (option, criterion) pair to ground the LLM's scoring in real evidence.

## Workflow

1. **Define** — Enter a decision goal, options (with optional PDF documents each), and weighted criteria.
2. **Research** — The AI Research Agent scores every (option, criterion) pair as triangular fuzzy numbers (l, m, u) with justifications. When PDFs are uploaded, RAG context is injected so scores are evidence-driven.
3. **Review** — Inspect and edit the AI-generated fuzzy scores and confirm criteria classification (benefit / cost).
4. **Calculate** — Deterministic Fuzzy TOPSIS computes closeness coefficients and ranks all options.
5. **Results** — View the winner, closeness coefficients, and a two-part explanation (Algorithmic Breakdown + Contextual Fit).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Pandas |
| Backend API | Django, Django CORS Headers |
| Data Models | Pydantic v2 |
| AI Orchestration | LangGraph |
| LLM | Groq (Llama 3.3 70B) via LangChain |
| RAG Embeddings | sentence-transformers (all-MiniLM-L6-v2), ChromaDB |
| PDF Processing | PyPDF, LangChain Text Splitters |

## Setup

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/SJSIO/Decison_Companion.git
cd Decison_Companion

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or export directly:

```
GROQ_API_KEY=gsk_your_key_here
```

Optional RAG configuration:

```
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2    # HuggingFace model id (default)
RAG_EMBEDDING_DEVICE=cpu                  # cpu or cuda (default: cpu)
```

### Run

Start both servers (two terminals):

```bash
# Terminal 1 — Django backend
python manage.py runserver

# Terminal 2 — Streamlit frontend
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in Terminal 2 (default: `http://localhost:8501`).

## My understanding of the problem.

From the time assignment was provided I thought of building  a domain specific decision making system but the session held in google meet changed my mind. I understood that the I am required to build a Global decision making system, not a domain specific one. Being a general decision making system, picking between complex stuff—like best car or hiring candidates—is super hard. Regular decision-making tools force you to manually type in every single score for each criterions provided by the user , which is boring and biased.But just throwing it at a basic llm wrapper doesn't work either because they have a chance to hallucinate and makes the project entirely depend on llm, so I made the agents to score each criterions based on options provided in the scale of 10 and user can also edit the scores if required.

I built this project to bridge that gap of llm scoring options which llm would have no knowledge of.So I implemented Rag as it reads real unstructured data (like pdfs), pulls out the actual facts, handles the uncertainty using fuzzy logic, and then uses  math (fuzzy topsis) to rank the options and actually explain why the winner won.


## Assumptions I made.

1.Fuzzy numbers: When I researched for the best algorithms decided to implement the fuzzy topsis algorithm on the assumption that generating three scores for each criterion a lower bond score,a middle score and an upper bond score made the decsion making more better.

2.I also assumed that the taking scores for the criteria's from the user is like asking for too many inputs and made llm to score options on the belief that llm's have knowledge enough to score these options.

3.Switching from normal Wiegh Sum Model to fuzzy topsis algorithm made the decision making better even though it may have some drawbacks

4.Implemented Rag to make decisions on topics that llms would have no knowledge about and when the user uploads files, where llm would have knowledge about, I made the assumption that the user would upload files which would make the decision making better not bad.

5.Through number of rag based test decision run I understood that modern llms suffer from "cautious alignment". if they don't see an exact keyword, they sometimes don't guess the scores. I assumed I'd need aggressive chain-of-thought prompting so the ai actually infers things like a human analyst would.



## Why i structured it this way?

My past projects were built on python so I choose python as the main programming language. You might look at this and think django plus langgraph is total overkill. but I used python Django as backend because developing like an api I would be easily able to change the frontend stack whenever required. My experience in frontend is quite low so I choose streamlit a simple web interface by python.

For the ai side, langgraph is essential. The rag pipeline here isn't just a simple prompt chain. To get the math right for fuzzy topsis, the research agent needs to loop and think. Langgraph handles that orchestration perfectly and sets up the architecture so users can drop in self-correcting agents later. I also split the streamlit app into multiple pages to keep the main input ui clean, while pushing all the heavy step-by-step math breakdowns to a separate page.

## Design decisions and trade-offs

 ###Option-specific vs Global rag: 
I made it so you upload pdfs for specific options instead of one giant upload bin. The ai literally can't mix up candidate a's resume with candidate b's because the retrieval is filtered by metadata tags.
 
###Semantic inference vs Hardcoded python: 
I ripped out the brittle python string-matching guardrails I tried first and replaced them with llm query expansion. it uses a bit more tokens, but it actually understands meaning now (like knowing "public relations" means good "communication") instead of failing just because it didn't find the exact word.

###Human-in-the-loop classification: 
I pause the app to let the user confirm if a criteria is a "cost" or "benefit". it adds a step, but it stops the algorithm from totally bombing an option just because the ai guessed the type wrong.

###Fuzzy triangular numbers vs. Simple 1-10 scoring: 

* Decision: I forced the llm to output three floats (lower_bound, most_likely, upper_bound) instead of just picking a single integer out of thin air.

trade-off: The python math pipeline for fuzzy topsis is way harder to build than a simple weighted average (you have to calculate normalized matrices and vertex distances). but it is mathematically honest. if the llm is unsure, it can widen the gap between the lower and upper bounds, letting the algorithm handle the uncertainty instead of hallucinating a fake absolute number.

###Decoupled django api vs. Monolith streamlit app: 

* Decision: I physically separated the ui from the ai logic.

trade-off: deploying two separate servers (frontend and backend) is definitely more annoying than deploying a single streamlit script. but streamlit re-runs its entire file on almost every user click. if i kept the langgraph logic inside streamlit, checking a single checkbox would accidentally re-trigger 30 seconds of expensive ai generation. the decoupled api acts as a firewall against that.


##Edge Cases Considered

###The empty context fallback: 
If you compare option a (with a pdf) and option b (no pdf), the app doesn't crash. the prompt dynamically tells the llm to just use its internet brain to score option b.

###Lazy llm defaults: 
To stop the ai from giving the exact same "insufficient data" excuse for everything, the prompt forces it to extract evidence first and literally bans fallback phrases.

###Rewarding actual data: 
I made sure options with real, documented evidence mathematically outrank the ones where the ai just had to guess.

###User Inputs:
I made sure that the user enters minimum of two options and one criteria otherwise the decision making would not run.

 ###LLm hallucinating the criteria type: 
Sometimes the ai gets confused by context. For example, it might classify "latency" as a benefit (thinking more is better) instead of a cost (where lower is better). I handled this edge case by pausing the graph before the math runs. letting the user manually override the cost/benefit classification ensures the algorithm doesn't mathematically punish the winning option due to a bad ai guess.


## What I would improve with more time?

###Deploy the critic agent: 
The langgraph setup is already there for it. I'd add a critic node to review the research agent's fuzzy scores against the pdf context and force a retry if it catches a hallucination.

###Database storage: 
I'd use the django sqlite/postgres setup to let users actually save their sessions, generate shareable links for the decision reports, and track stuff over time.

###Live web search rag: 
I'd hook up something like tavily or serpapi. that way, if a user doesn't upload a pdf, the ai can actively search the live web for the newest pricing or specs instead of relying on old training data.

###More optimized UI
For a first time user it would take time for the user to interrupt the page and find out the things that is needed to be inputted. By optimizing the frontend it would help users to run decsions more faster

## Project Structure

```
Decsion_Companion/
├── config/                  # Django project settings & URL config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── decision_comp/           # Django app — core logic
│   ├── models.py            # Pydantic schemas (options, criteria, scores, TFN)
│   ├── views.py             # API endpoints (/api/research/, /api/calculate/)
│   ├── graph.py             # LangGraph agents, Fuzzy TOPSIS implementation
│   ├── llm_services.py      # LLM prompts, structured output, deterministic guardrails
│   ├── rag.py               # PDF loading, chunking, embedding, ChromaDB retrieval
│   ├── urls.py              # API URL routing
│   └── tests/               # Unit tests (API research, RAG)
├── pages/
│   └── How_this_Algorithm_works.py   # Streamlit page explaining Fuzzy TOPSIS
├── streamlit_app.py         # Main Streamlit frontend (5-phase workflow)
├── manage.py                # Django management
├── requirements.txt         # Python dependencies
├── BUILD_PROCESS.md         # Development journal & design decisions
└── RESEARCH_LOG.md          # Research notes
```

## API Endpoints

### POST `/api/research/`

Runs the AI Research Agent to score all (option, criterion) pairs.

**Request body:**
```json
{
  "problem_description": "Hiring the best candidate",
  "options": [
    {
      "name": "Candidate A",
      "description": "Software developer",
      "documents": [{ "filename": "resume.pdf", "content_base64": "..." }]
    }
  ],
  "criteria": [
    { "name": "Communication", "weight": 8, "description": "...", "kind": "benefit" }
  ]
}
```

**Response:** Array of fuzzy scores (l, m, u) with justifications, plus classified criteria metadata.

### POST `/api/calculate/`

Runs Fuzzy TOPSIS on provided scores and returns the ranked result with an explanation.

**Request body:** Same structure plus a `scores` array of `{ option_name, criterion_name, l, m, u, justification }`.

**Response:** Winner, loser, closeness coefficients per option, explanation, and calculation intermediates.

## How Fuzzy TOPSIS Works

1. **Normalize** — Raw fuzzy scores are normalized per criterion type (benefit: divide by max; cost: invert and scale).
2. **Weight** — Each criterion's weight (1–10) is applied to the normalized matrix.
3. **Ideal Solutions** — The Fuzzy Positive Ideal Solution (FPIS) and Fuzzy Negative Ideal Solution (FNIS) are computed from the weighted matrix.
4. **Distances** — Each option's distance to FPIS and FNIS is calculated using the vertex method.
5. **Closeness Coefficient** — CC = distance_to_FNIS / (distance_to_FPIS + distance_to_FNIS). Higher CC means closer to the ideal.
6. **Rank** — Options are ranked by CC; the highest wins.

## RAG Pipeline

When PDFs are uploaded per option:

1. Each PDF is loaded and split into overlapping text chunks.
2. Chunks are embedded locally using a HuggingFace SentenceTransformer model.
3. Chunks are stored in an ephemeral ChromaDB collection with `option_name` metadata.
4. For each (option, criterion) pair, a criterion-specific similarity search retrieves the most relevant chunks (filtered by option).
5. Retrieved context is annotated with keyword-match markers and injected into the LLM prompt.
6. Deterministic guardrails boost scores when keyword evidence is found and ensure evidence-bearing options outscore those without evidence.

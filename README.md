# Data Analytics Assistant

An AI-powered analytics agent that lets you explore tabular data through natural language. Ask questions in plain English — no pandas syntax required. The agent writes and executes Python code using a **ReAct loop** (Reason → Act → Observe), maintains conversation memory across follow-up questions, and automatically creates visualizations based on the type of question asked.

Built as part of the Open Avenues Build Fellowship: AI-Powered Data Exploration program.

---

## Features

- **Natural language queries** — ask questions about your data without writing any code
- **ReAct loop** — multi-step reasoning where the agent thinks, writes code, observes the result, and repeats until it has a complete answer
- **Code generation and execution** — the agent generates real Python code and runs it in a sandboxed environment
- **Conversation memory** — ask follow-up questions; the agent remembers previous answers using a sliding-window memory with automatic summarization
- **Auto-visualization** — the agent automatically creates charts (bar, line, scatter, histogram, pie, box) based on the type of question, even when not explicitly asked
- **14 analytics tools** — filter, group, aggregate, sort, compute stats, load files, and 6 chart types
- **Safe code execution** — AST validation blocks imports and dangerous operations before any code runs
- **Included dataset** —  Sleep Health data (100,000 records, 32 columns)

---

## Project Structure

```
data-analytics-assistant/
├── notebooks/
│   └── sleep_health_assistant.ipynb    # Main demo notebook
├── src/
│   ├── agent.py                        # ReAct agent — orchestration loop, code execution, synthesis
│   ├── tools.py                        # 14 analytics tools (filter, group, charts, load, etc.)
│   ├── memory.py                       # Sliding-window conversation memory + summarization
│   └── __init__.py
├── web/
│   ├── server.py                       # FastAPI backend — chat, file upload, session reset
│   └── static/
│       ├── index.html                  # Chat UI
│       ├── app.js                      # Frontend logic
│       └── style.css                   # Styles
├── data/
│   └── sleep_health_dataset.csv        # Sleep health and lifestyle dataset (100k records)
├── pyproject.toml                      # Project dependencies
├── requirements.txt                    # Pip-compatible dependency list
├── .env                                # Your API key — not committed to git
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd data-analytics-assistant
```

### Step 2 — Install dependencies

**Option A — using pip (recommended if you don't have uv):**

```bash
pip install -r requirements.txt
```

**Option B — using uv:**

```bash
pip install uv
uv sync
```

### Step 3 — Configure your API key

Create a `.env` file in the `data-analytics-agent/` folder:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

### Step 4 — Launch JupyterLab

```bash
jupyter lab
```

Then open the notebook from the `notebooks/` folder.

---

## How to Run the Agent

### Option 1 — Web UI 

Start the FastAPI server from the project root:

```bash
uvicorn web.server:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

The Sleep Health dataset loads automatically. From there you can:
- Type questions in the chat input and hit Enter (or click Send)
- Upload your own CSV, Excel, or Parquet file using the sidebar
- Reset the session (clears memory and uploaded datasets) with the Reset button

> Make sure you run the command from the project root, not inside the `web/` folder.

### Option 2 — Run interactively via notebook

Open `notebooks/sleep_health_assistant.ipynb` and run all cells from top to bottom:

1. **Setup cell** — loads your `.env` and imports the agent
2. **Load dataset cell** — loads the CSV into a pandas DataFrame
3. **Create agent cell** — initializes `DataAnalyticsAgent` with your data
4. **Question cells** — each cell runs one question through the ReAct loop
5. **Interactive chat loop** — the last cell lets you type questions freely

```python
# Example: create the agent and ask a question
agent = DataAnalyticsAgent(
    client=client,
    tables={"sleep": df},
    model="gpt-4o-mini",
    max_steps=10,
)

answer = agent.run("Which mental health condition is associated with the worst average sleep quality?")
print(answer)
```

---

## Example Queries

**Exploration**
```
Give me an overview of this dataset — columns, data types, and summary statistics
Show me the first few rows of the data
```

**Comparisons (auto-generates bar chart)**
```
Which occupation has the highest average sleep quality score?
How does stress score differ between shift workers and non-shift workers?
Which season is associated with the best sleep duration?
```

**Distributions (auto-generates histogram or box plot)**
```
Show me the distribution of sleep latency across the dataset
How spread out are cognitive performance scores?
```

**Relationships (auto-generates scatter plot)**
```
Is there a relationship between screen time before bed and sleep quality?
Does BMI correlate with sleep duration?
```

**Multi-step analysis**
```
Among people with severe sleep disorder risk, what is their average stress score and caffeine intake?
What are the top 5 occupations with the highest average stress score?
```

**Multi-turn conversation**
```
# Turn 1
Which mental health condition has the worst average sleep quality?

# Turn 2 (agent remembers the previous answer)
For that group, what is their average stress score and caffeine intake compared to healthy people?
```

---

## How the ReAct Loop Works

Each call to `agent.run()` runs a loop of up to 10 steps:

1. **Reason** — the LLM receives the question, all available tools, and every observation collected so far. It returns a structured `AgentStep` with a `thought`, `code`, and `is_done` flag.
2. **Act** — the generated code is validated (AST check) then executed in a shared namespace where all tool functions are available.
3. **Observe** — the output (stdout + last expression value) becomes the observation, appended to the list for the next step.
4. **Repeat** until `is_done=True` or max steps reached.
5. **Synthesize** — a separate LLM call turns all observations into a clean final answer.

Variables assigned in one step persist into the next (like notebook cells), so the agent can chain operations across steps.

---

## Datasets

### Sleep Health Dataset (`data/sleep_health_dataset.csv`)
100,000 records, 32 columns covering sleep patterns, lifestyle habits, and health outcomes. Key columns:
- Sleep metrics: `sleep_duration_hrs`, `sleep_quality_score`, `rem_percentage`, `deep_sleep_percentage`, `sleep_latency_mins`
- Lifestyle: `stress_score`, `exercise_day`, `caffeine_mg_before_bed`, `screen_time_before_bed_mins`
- Health: `mental_health_condition`, `sleep_disorder_risk`, `bmi`, `heart_rate_resting_bpm`
- Demographics: `age`, `gender`, `occupation`, `country`, `chronotype`

The dataset is included in the repository under `data/`. You can also substitute any CSV, Excel, or Parquet file using the `load_csv` tool.

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

| Library | Version | Purpose |
|---|---|---|
| `openai` | >=1.60 | LLM API calls and structured output parsing |
| `pandas` | >=2.0 | Data manipulation |
| `numpy` | >=1.26 | Numerical operations |
| `matplotlib` | >=3.8 | Chart generation |
| `pydantic` | >=2.0 | Structured LLM output models |
| `python-dotenv` | >=1.0 | Loading API key from `.env` |
| `fastapi` | >=0.110 | Web UI backend |
| `uvicorn` | >=0.29 | ASGI server for FastAPI |
| `python-multipart` | >=0.0.9 | File upload support |
| `openpyxl` | >=3.1 | Excel file support |
| `jupyter` | >=1.0 | Notebook interface |
| `ipykernel` | >=6.0 | Jupyter kernel |

---

"""
ReAct agent core — orchestrates the Thought → Code → Observation loop.

Architecture:
  1. User sends a question
  2. Agent builds a prompt with dataset context + tool descriptions
  3. LLM returns a structured AgentStep (thought + code + is_done flag)
  4. Code is executed in a sandboxed environment
  5. Observation is fed back; loop repeats until is_done=True or max_steps reached
  6. A final synthesis call turns the observations into a clean answer
"""

from __future__ import annotations

import ast
import io
import contextlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

from src.memory import ConversationMemory
from src.tools import AnalyticsTools


# ─────────────────────────────────────────────────────────────
# Structured LLM output for one ReAct step
# ─────────────────────────────────────────────────────────────

class AgentStep(BaseModel):
    """One step in the ReAct loop."""
    thought: str          # Agent's reasoning about what to do next
    code: str             # Python code to execute (empty string if done)
    is_done: bool         # True when the agent has enough to answer


# ─────────────────────────────────────────────────────────────
# Safe code executor
# ─────────────────────────────────────────────────────────────

# AST node types that are not allowed in generated code
_DISALLOWED_NODES = (
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Global, ast.Nonlocal,
)

_DISALLOWED_NAMES = {
    "open", "exec", "eval", "compile", "__import__",
    "os", "sys", "subprocess", "pathlib", "shutil",
    "socket", "requests", "http", "urllib",
}


def _validate_code(code: str) -> tuple[bool, str]:
    """Return (ok, error_message). Rejects unsafe AST nodes and names."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, _DISALLOWED_NODES):
            return False, f"Disallowed syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in _DISALLOWED_NAMES:
            return False, f"Disallowed name: '{node.id}'"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"Disallowed dunder attribute: '{node.attr}'"

    return True, "ok"


def _execute_code(code: str, exec_env: dict[str, Any]) -> tuple[str, str]:
    """
    Run code in exec_env. Returns (stdout_output, error_message).
    error_message is empty on success.
    Automatically captures the value of the last expression (like a notebook cell).
    """
    ok, err = _validate_code(code)
    if not ok:
        return "", err

    # Rewrite the last bare expression as an assignment so we can capture it
    last_value = None
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1]
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id="_last_expr_", ctx=ast.Store())],
                value=last_expr.value,
                lineno=last_expr.lineno,
                col_offset=last_expr.col_offset,
            )
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
    except Exception:
        pass  # if rewrite fails, run original code as-is

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_env)  # noqa: S102
        last_value = exec_env.pop("_last_expr_", None)
        printed = stdout_buf.getvalue().strip()

        parts = []
        if printed:
            parts.append(printed)
        if last_value is not None:
            parts.append(repr(last_value))

        return "\n".join(parts) if parts else "", ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

@dataclass
class DataAnalyticsAgent:
    """
    Conversational data analytics agent using a ReAct loop.

    Usage:
        agent = DataAnalyticsAgent(client, tables={"sales": df})
        answer = agent.run("What are the top 5 products by revenue?")
    """

    client: OpenAI
    tables: dict[str, pd.DataFrame]
    model: str = "gpt-4o-mini"
    max_steps: int = 10
    memory: ConversationMemory = field(default_factory=ConversationMemory)

    def __post_init__(self) -> None:
        self.tools = AnalyticsTools(self.tables)

    def run(self, question: str, verbose: bool = True) -> str:
        """
        Run the ReAct loop for one user question.
        Returns the final synthesized answer as a string.
        """
        self.memory.add_user(question)

        observations: list[str] = []
        # Shared execution environment — variables persist across all steps
        # within one run() call, just like notebook cells
        exec_env = self._make_exec_env()

        for step_num in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"Step {step_num}")
                print("─" * 60)

            # Rebuild system prompt each step so newly loaded tables are visible
            system_prompt = self._build_system_prompt()

            # Build messages: system + memory + current observations
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.memory.get_messages())

            if observations:
                obs_text = "\n\n".join(
                    f"Observation {i + 1}:\n{o}" for i, o in enumerate(observations)
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Previous observations:\n{obs_text}\n\n"
                        "Continue the analysis or set is_done=true if you have enough to answer."
                    ),
                })
            else:
                messages.append({"role": "user", "content": f"Question: {question}"})

            # Ask the LLM for the next step
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=AgentStep,
                temperature=0,
            )
            step: AgentStep = response.choices[0].message.parsed

            if verbose:
                print(f"Thought: {step.thought}")
                if step.code:
                    print(f"Code:\n{step.code}")

            # Execute the code if any
            if step.code.strip():
                # Sync any tables added in previous steps into exec_env
                # so save_as results are accessible as Python variables
                for tname, tdf in self.tools.tables.items():
                    if tname not in exec_env:
                        exec_env[tname] = tdf

                output, error = _execute_code(step.code, exec_env)

                # Sync newly saved tables back into exec_env after execution
                for tname, tdf in self.tools.tables.items():
                    exec_env[tname] = tdf

                if error:
                    observation = f"Error: {error}"
                    if verbose:
                        print(f"Observation: {observation}")
                else:
                    observation = output or "(code ran successfully, no output)"
                    if verbose:
                        print(f"Observation: {observation}")

                observations.append(f"Thought: {step.thought}\nCode: {step.code}\nResult: {observation}")
            else:
                observations.append(f"Thought: {step.thought}")

            if step.is_done:
                break
        else:
            observations.append("Reached maximum steps.")

        # Synthesize a clean final answer
        answer = self._synthesize(question, observations)
        self.memory.add_assistant(answer)
        self.memory.maybe_summarize(self.client, self.model)
        return answer

    # ── Private helpers ───────────────────────────────────────

    def _build_system_prompt(self) -> str:
        table_descriptions = []
        for name, df in self.tables.items():
            cols = ", ".join(df.columns.tolist())
            table_descriptions.append(
                f"  - {name}: {df.shape[0]} rows × {df.shape[1]} columns\n"
                f"    Columns: {cols}"
            )
        tables_text = (
            "\n".join(table_descriptions)
            if table_descriptions
            else "  (no tables loaded yet — use load_csv() to load a file)"
        )

        return f"""You are a data analytics assistant. Answer questions about data by writing Python code that calls the available tools.

## Loaded Tables
{tables_text}

## Available Tools
{self.tools.describe_all()}

## How to use tools
- Load new data: load_csv("../data/file.csv", table_name="sales")
- Explore: result = describe("sales"); print(result)
- Count rows per category: result = count_by("sleep", "sleep_disorder_risk", save_as="risk_counts"); print(result)
- Store intermediate results: filter_rows("sales", "region", "==", "West", save_as="west")
- Chain operations: group_and_agg("west", "category", "revenue", "sum", save_as="west_rev")
- Print results you want to observe: print(result)
- For percentage/proportion questions: use count_by to get counts per group, then compute the percentage from the result rows

## Visualization rules — follow these automatically
- Comparison across categories (e.g. "by region", "by category", "which is highest") → plot_bar
- Trend over time or ordered sequence → plot_line
- Relationship between two numeric columns → plot_scatter
- Distribution of a single column → plot_histogram or plot_box
- Part-to-whole (e.g. "share", "percentage", "breakdown") → plot_pie
- Comparing distributions across groups → plot_box
- ALWAYS create a chart when the question involves comparing, ranking, trending, or distributing data — even if the user did not explicitly ask for one

## Rules
- ALWAYS call print() on every tool result — without print(), you get no observation and cannot reason about the data
- Write one focused operation per step
- Use describe() or preview() first if you're unsure about column names
- Set is_done=true only when you have enough observations to answer the question
- Do not import libraries — pd and np are available directly

## Examples of correct usage
```python
# CORRECT — result is printed so you can see it
result = group_and_agg("sleep", "mental_health_condition", "sleep_quality_score", "mean")
print(result)

# CORRECT — preview is printed
print(preview("sleep", n=3))

# WRONG — no print means no observation, agent is blind
result = group_and_agg("sleep", "mental_health_condition", "sleep_quality_score", "mean")
```
"""

    def _make_exec_env(self) -> dict[str, Any]:
        """Build the execution namespace for generated code."""
        import numpy as np

        env: dict[str, Any] = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "list": list,
                "dict": dict, "str": str, "int": int, "float": float,
                "bool": bool, "abs": abs, "round": round, "sum": sum,
                "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "isinstance": isinstance, "type": type,
            },
            "pd": pd,
            "np": np,
        }
        # Inject all tool methods directly into the namespace
        for name in self.tools.ACTION_NAMES:
            env[name] = getattr(self.tools, name)

        return env

    def _synthesize(self, question: str, observations: list[str]) -> str:
        """Ask the LLM to write a clean final answer from the observations."""
        obs_text = "\n\n".join(observations)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analytics assistant. "
                        "Write a clear, concise answer to the user's question "
                        "based on the analysis observations. "
                        "Be specific — include numbers, rankings, and key findings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Analysis observations:\n{obs_text}\n\n"
                        "Write the final answer:"
                    ),
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

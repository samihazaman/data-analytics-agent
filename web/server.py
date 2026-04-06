"""
FastAPI backend for the Data Analytics Assistant web UI.

Run with:
    uvicorn web.server:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI popups
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from src.agent import DataAnalyticsAgent

load_dotenv()

app = FastAPI(title="Data Analytics Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global agent state ────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_DATA = Path("data/sleep_health_dataset.csv")

def _make_agent(tables: dict[str, pd.DataFrame]) -> DataAnalyticsAgent:
    return DataAnalyticsAgent(client=client, tables=tables, model="gpt-4o-mini", max_steps=10)

def _load_default() -> dict[str, pd.DataFrame]:
    if DEFAULT_DATA.exists():
        df = pd.read_csv(DEFAULT_DATA)
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )
        return {"sleep": df}
    return {}

_tables: dict[str, pd.DataFrame] = _load_default()
_agent: DataAnalyticsAgent = _make_agent(_tables)


# ── Chart capture ─────────────────────────────────────────────

def _make_show_capture(captured: list[str]):
    """Return a plt.show() replacement that saves figures as base64 PNG strings."""
    def _show():
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        captured.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close("all")
    return _show


# ── Request / response models ─────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    images: list[str] = []

class TableInfo(BaseModel):
    name: str
    rows: int
    columns: list[str]

class TablesResponse(BaseModel):
    tables: list[TableInfo]


# ── Routes ────────────────────────────────────────────────────

@app.get("/api/tables", response_model=TablesResponse)
def list_tables() -> Any:
    return {
        "tables": [
            {"name": name, "rows": len(df), "columns": df.columns.tolist()}
            for name, df in _agent.tools.tables.items()
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> Any:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not _agent.tools.tables:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please upload a file first.")
    try:
        captured_images: list[str] = []
        # Patch plt.show in the agent's exec env to capture instead of display
        original_show = plt.show
        plt.show = _make_show_capture(captured_images)
        try:
            answer = _agent.run(req.question, verbose=False)
        finally:
            plt.show = original_show

        return {"answer": answer, "images": captured_images}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> Any:
    global _agent, _tables

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".parquet"}:
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .parquet files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".csv":
            df = pd.read_csv(tmp_path)
        elif suffix == ".xlsx":
            df = pd.read_excel(tmp_path)
        else:
            df = pd.read_parquet(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    # Normalize column names
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    import re
    table_name = re.sub(r"[^a-zA-Z0-9]+", "_", Path(file.filename).stem.lower()).strip("_")
    _tables[table_name] = df

    # Rebuild agent with updated tables, preserving memory
    old_memory = _agent.memory
    _agent = _make_agent(_tables)
    _agent.memory = old_memory

    return {"table_name": table_name, "rows": len(df), "columns": df.columns.tolist()}


@app.post("/api/reset")
def reset() -> Any:
    global _agent, _tables
    _tables = _load_default()
    _agent = _make_agent(_tables)
    return {"status": "reset", "tables": list(_tables.keys())}


# ── Serve static files ────────────────────────────────────────

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_static / "index.html"))

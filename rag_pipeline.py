"""
rag_pipeline.py
───────────────
RAG pipeline powered by Gemini via the google.genai SDK.

IMPORTANT — model name changes as of 2026:
  - gemini-1.5-flash, gemini-1.5-flash-8b  → deprecated / unreliable
  - gemini-2.0-flash, gemini-2.0-flash-lite → SHUT DOWN June 1 2026
  - gemini-2.5-flash-lite                   → FREE tier, use this
  - gemini-2.5-flash                         → FREE tier fallback
"""

import os
import json
import asyncio
from collections import Counter, deque

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages  import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from google import genai as _genai
from google.genai import types as _genai_types

from fastapi           import APIRouter
from fastapi.responses import JSONResponse
from pydantic          import BaseModel

load_dotenv(override=False)

DATA_DIR = os.getenv("DATA_DIR", ".")

# ── Model selection ───────────────────────────────────────────────────────────
# gemini-2.5-flash-lite  → free tier, fastest, use this as default
# gemini-2.5-flash       → free tier, smarter, used as fallback
# DO NOT use 1.5-flash-8b, 2.0-flash — deprecated/shut down in 2026
GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"

print(f"  RAG using model: {GEMINI_MODEL}")

# ── In-memory stores ──────────────────────────────────────────────────────────
_raw_documents: dict = {}
_rag_chains:    dict = {}
_histories:     dict = {}
_init_errors:   dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_parquet_docs(parquet_path: str, season: str, year: int) -> list:
    df = pd.read_parquet(parquet_path)

    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(
            lambda x: list(x) if isinstance(x, (set, list)) else []
        )
    else:
        df["skills_found"] = [[] for _ in range(len(df))]

    # Re-extract skills if >80% of rows are empty
    empty_pct = df["skills_found"].apply(lambda x: len(x) == 0).mean()
    if empty_pct > 0.8:
        print(f"  {empty_pct:.0%} rows have no skills — re-extracting...")
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_generator import extract_skills as _extract, normalize_text as _norm
        full_text = (
            df.get("description", pd.Series("", index=df.index)).fillna("") + " " +
            df.get("highlights",  pd.Series("", index=df.index)).fillna("")
        )
        df["skills_found"] = full_text.apply(lambda t: _extract(_norm(t)))
        n = df["skills_found"].apply(lambda x: len(x) > 0).sum()
        print(f"  Re-extracted: {n}/{len(df)} jobs now have skills")

    docs = []

    for _, row in df.head(100).iterrows():
        skills = ", ".join(row["skills_found"]) if row["skills_found"] else "not specified"
        desc   = str(row.get("description", ""))[:300]
        text   = (
            f"Job Title: {row.get('title', 'Unknown')}\n"
            f"Company: {row.get('company', 'Unknown')}\n"
            f"Location: {row.get('location_city', 'Unknown')}, {row.get('location_state', '')}\n"
            f"Remote: {'Yes' if row.get('is_remote') else 'No'}\n"
            f"Skills Required: {skills}\n"
            f"Apply Link: {row.get('apply_link', '')}\n"
            f"Description: {desc}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "title":   str(row.get("title",   "")).lower(),
                "company": str(row.get("company", "")).lower(),
                "type":    "job",
                "season":  season,
                "year":    year,
            }
        ))

    # Aggregate summary document
    dedup_cols = [c for c in ["title", "company", "apply_link"] if c in df.columns]
    unique_df  = df.drop_duplicates(subset=dedup_cols) if dedup_cols else df
    total      = len(unique_df)
    remote_pct = round(unique_df["is_remote"].mean() * 100, 1) if total else 0
    top_cities = unique_df["location_city"].value_counts().head(5).to_dict()
    all_skills = [s for row in unique_df["skills_found"] for s in row]
    top_skills = [s for s, _ in Counter(all_skills).most_common(20)]
    role_dist  = (
        df["role_category"].value_counts().head(8).to_dict()
        if "role_category" in df.columns else {}
    )

    summary = (
        f"DATASET SUMMARY — {season} {year} Co-op Jobs\n"
        f"Total unique jobs: {total}\n"
        f"Remote jobs: {remote_pct}%\n"
        f"Top hiring cities: {json.dumps(top_cities)}\n"
        f"Top 20 most demanded skills: {', '.join(top_skills)}\n"
        f"Role category distribution: {json.dumps(role_dist)}\n"
        f"Unique companies: {df['company'].nunique()}\n"
        f"Scraped dates: {sorted(df['scraped_date'].unique().tolist()) if 'scraped_date' in df.columns else []}\n"
    )
    docs.append(Document(
        page_content=summary,
        metadata={"type": "summary", "season": season, "year": year}
    ))

    print(f"  Built {len(docs)} documents for {season} {year}")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are Co-operator AI, an expert career assistant embedded in the Co-operator "
    "platform — a real-time co-op job intelligence tool for Canadian university students.\n\n"
    "You have TWO knowledge sources — use BOTH:\n"
    "1. Live {season} {year} co-op job data (retrieved context below)\n"
    "2. Your own general knowledge about careers, skills, and learning resources\n\n"
    "Guidelines:\n"
    "- Be concise but informative. Use bullet points for lists.\n"
    "- When citing specific jobs, mention the company and title.\n"
    "- If the user asks for job counts or stats, give exact numbers from the data.\n"
    "- When asked about skills, ALWAYS use the DATASET SUMMARY section.\n"
    "- Individual job docs may say 'Skills Required: not specified' — ignore those;\n"
    "  use the summary aggregate for skill-related questions.\n"
    "- For HOW TO LEARN a skill: answer freely using your general knowledge.\n"
    "- For salaries, interview prep, resume tips: use your general knowledge.\n"
    "- Only decline if the question is completely unrelated to careers or jobs.\n\n"
    "Retrieved job context:\n{context}"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chain(season: str, year: int, api_key: str) -> RunnableLambda:
    client      = _genai.Client(api_key=api_key)
    system_text = _SYSTEM.replace("{season}", season).replace("{year}", str(year))

    def _retrieve(inputs: dict) -> dict:
        key      = f"{season}_{year}"
        docs     = _raw_documents.get(key, [])
        summary  = [d for d in docs if d.metadata.get("type") == "summary"]
        job_docs = [d for d in docs if d.metadata.get("type") == "job"]
        q_lower  = inputs["question"].lower()
        matched  = []
        for d in job_docs:
            words = [w for w in q_lower.split() if len(w) > 4]
            if (d.metadata.get("title", "") in q_lower or
                    d.metadata.get("company", "") in q_lower or
                    any(w in d.page_content.lower() for w in words)):
                matched.append(d)
                if len(matched) >= 8:
                    break
        if not matched:
            matched = job_docs[:5]
        inputs["context"] = "\n\n---\n\n".join(
            d.page_content for d in summary + matched
        )
        return inputs

    def _generate(inputs: dict) -> str:
        contents = []
        for msg in inputs.get("chat_history", []):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "model"
            contents.append(
                _genai_types.Content(
                    role=role,
                    parts=[_genai_types.Part.from_text(text=msg.content)]
                )
            )

        final_query = (
            f"Retrieved job context:\n{inputs['context']}\n\n"
            f"User Question: {inputs['question']}"
        )
        contents.append(
            _genai_types.Content(
                role="user",
                parts=[_genai_types.Part.from_text(text=final_query)]
            )
        )

        cfg = _genai_types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=0.3,
            max_output_tokens=800,
        )

        # Try primary model, fall back if unavailable
        models_to_try = [GEMINI_MODEL]
        if GEMINI_MODEL != GEMINI_MODEL_FALLBACK:
            models_to_try.append(GEMINI_MODEL_FALLBACK)

        last_err = None
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=cfg,
                )
                return response.text
            except Exception as e:
                last_err = e
                err_str  = str(e)
                # Only try fallback on model-availability errors
                if any(code in err_str for code in ["403", "404", "invalid", "not found",
                                                     "MODEL_NOT_FOUND", "deprecated"]):
                    print(f"  Model {model} unavailable — trying {GEMINI_MODEL_FALLBACK}...")
                    continue
                raise  # re-raise rate limit / auth errors immediately

        raise last_err

    def _chain(inputs: dict) -> str:
        return _generate(_retrieve(inputs))

    return RunnableLambda(_chain)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def init_rag(season: str, year: int) -> bool:
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    key     = f"{season}_{year}"
    path    = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")

    if not os.path.exists(path):
        msg = f"No parquet found at '{path}'. Scrape {season} {year} first."
        print(f"  {msg}")
        _init_errors[key] = msg
        return False

    if not api_key:
        msg = "GOOGLE_API_KEY is not set in Railway environment variables."
        print(f"  {msg}")
        _init_errors[key] = msg
        return False

    try:
        print(f"  RAG: initialising for {key} using {GEMINI_MODEL}...")
        if key not in _raw_documents:
            _raw_documents[key] = _load_parquet_docs(path, season, year)
        _rag_chains[key] = _build_chain(season, year, api_key)
        _histories[key]  = deque(maxlen=10)
        _init_errors.pop(key, None)
        print(f"  RAG ready for {key}")
        return True
    except Exception as e:
        import traceback
        msg = f"RAG init error: {e}"
        print(f"  {msg}\n{traceback.format_exc()}")
        _init_errors[key] = msg
        return False


def ask(question: str, season: str, year: int) -> str:
    key = f"{season}_{year}"
    if key not in _rag_chains:
        if not init_rag(season, year):
            return f"Error: {_init_errors.get(key, 'RAG not initialised.')}"

    try:
        answer = _rag_chains[key].invoke({
            "question":     question,
            "chat_history": list(_histories[key]),
        })
        _histories[key].append(HumanMessage(content=question))
        _histories[key].append(AIMessage(content=answer))
        return answer

    except Exception as e:
        err = str(e)
        print(f"  Gemini error (model={GEMINI_MODEL}): {err}")

        if any(x in err for x in ["API_KEY", "api key", "401", "PERMISSION_DENIED"]):
            return (
                "Authentication error — your GOOGLE_API_KEY is invalid or missing. "
                "Check Railway environment variables."
            )

        if "403" in err:
            return (
                f"Access denied for model '{GEMINI_MODEL}'. "
                "Go to aistudio.google.com, generate a fresh API key, "
                "and update GOOGLE_API_KEY in Railway."
            )

        if any(x in err for x in ["404", "not found", "MODEL_NOT_FOUND", "deprecated"]):
            return (
                f"Model '{GEMINI_MODEL}' not found — it may have been deprecated. "
                "Set GEMINI_MODEL=gemini-2.5-flash-lite in Railway environment variables."
            )

        if any(x in err for x in ["429", "RESOURCE_EXHAUSTED", "quota"]):
            return (
                f"Rate limit reached on '{GEMINI_MODEL}' "
                f"(free tier: ~500 req/day). Try again tomorrow."
            )

        return f"AI error: {err[:300]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

chat_router = APIRouter(prefix="/api/chat", tags=["chatbot"])


class ChatRequest(BaseModel):
    question: str
    season:   str = "Summer"
    year:     int = 2026


class ChatResponse(BaseModel):
    answer: str
    season: str
    year:   int
    ready:  bool


@chat_router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    season = req.season.capitalize()
    if season not in ["Summer", "Fall", "Winter"]:
        return JSONResponse(status_code=400, content={"error": "Invalid season"})
    loop   = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask, req.question, season, req.year)
    return ChatResponse(
        answer=answer,
        season=season,
        year=req.year,
        ready=f"{season}_{req.year}" in _rag_chains,
    )


@chat_router.get("/status/{season}/{year}")
async def chat_status(season: str, year: int):
    season = season.capitalize()
    key    = f"{season}_{year}"
    return {
        "season": season,
        "year":   year,
        "ready":  key in _rag_chains,
        "model":  GEMINI_MODEL,
        "error":  _init_errors.get(key),
    }


@chat_router.post("/init/{season}/{year}")
async def chat_init(season: str, year: int):
    season = season.capitalize()
    loop   = asyncio.get_event_loop()
    ok     = await loop.run_in_executor(None, init_rag, season, year)
    return {
        "season":      season,
        "year":        year,
        "initialized": ok,
        "model":       GEMINI_MODEL,
        "error":       _init_errors.get(f"{season}_{year}"),
    }
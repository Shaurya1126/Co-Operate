"""
rag_pipeline.py
───────────────
LangChain v0.3 RAG pipeline (LCEL-based, no deprecated chains/memory imports)
powered by Gemini gemini-2.5-flash-lite.

Exposes a FastAPI router that plugs into api.py:
    from rag_pipeline import chat_router, init_rag as _rag_init
    app.include_router(chat_router)
"""

import os
import json
import asyncio
from collections import Counter, deque
import time
import pandas as pd
from dotenv import load_dotenv

# Load env variables early
load_dotenv()

# ── LangChain v0.3 imports (all from *-core / *-community / *-google-genai) ──
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

print(f"DEBUG GOOGLE_API_KEY = '{os.getenv('GOOGLE_API_KEY')}'")
DATA_DIR = os.getenv("DATA_DIR", ".")
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ── In-memory stores ──────────────────────────────────────────────────────────
_vector_stores: dict = {}
_rag_chains: dict = {}
_histories: dict = {}   # key → deque of LangChain message objects


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

    # If skills_found is empty for most rows, re-extract from description
    empty_pct = df["skills_found"].apply(lambda x: len(x) == 0).mean()
    if empty_pct > 0.8:
        print(f"  ⚠️  {empty_pct:.0%} of rows have no skills — re-extracting from descriptions...")
        import re as _re
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_generator import extract_skills as _extract, normalize_text as _norm, SKILLS as _SKILLS
        full_text = (
            df.get("description", pd.Series("", index=df.index)).fillna("") + " " +
            df.get("highlights", pd.Series("", index=df.index)).fillna("")
        )
        df["skills_found"] = full_text.apply(lambda t: _extract(_norm(t)))
        n_with_skills = df["skills_found"].apply(lambda x: len(x) > 0).sum()
        print(f"  ✅ Re-extracted skills: {n_with_skills}/{len(df)} jobs now have skills")

    docs = []

    for _, row in df.iterrows():
        skills = ", ".join(row["skills_found"]) if row["skills_found"] else "not specified"
        desc = str(row.get("description", ""))[:600]
        text = (
            f"Job Title: {row.get('title', 'Unknown')}\n"
            f"Company: {row.get('company', 'Unknown')}\n"
            f"Location: {row.get('location_city', 'Unknown')}, {row.get('location_state', '')}\n"
            f"Remote: {'Yes' if row.get('is_remote') else 'No'}\n"
            f"Season: {row.get('season', season)} {year}\n"
            f"Employment Type: {row.get('employment_type', 'Unknown')}\n"
            f"Skills Required: {skills}\n"
            f"Description (excerpt): {desc}\n"
            f"Apply Link: {row.get('apply_link', 'N/A')}\n"
            f"Posted: {str(row.get('scraped_date', 'Unknown'))}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={"title": str(row.get("title", "")), "company": str(row.get("company", "")),
                      "season": season, "year": year}
        ))

    # Aggregate summary document — deduplicate first so stats match the website
    dedup_cols = [c for c in ["title", "company", "apply_link"] if c in df.columns]
    unique_df = df.drop_duplicates(subset=dedup_cols) if dedup_cols else df
    total = len(unique_df)
    remote_pct = round(unique_df["is_remote"].mean() * 100, 1) if total else 0
    top_cities = unique_df["location_city"].value_counts().head(5).to_dict()
    all_skills = [s for row in unique_df["skills_found"] for s in row]
    top_skills = [s for s, _ in Counter(all_skills).most_common(20)]
    role_dist = (df["role_category"].value_counts().head(8).to_dict()
                  if "role_category" in df.columns else {})

    summary = (
        f"DATASET SUMMARY — {season} {year} Co-op Jobs\n"
        f"Total jobs scraped: {total}\n"
        f"Remote jobs: {remote_pct}%\n"
        f"Top hiring cities: {json.dumps(top_cities)}\n"
        f"Top 20 most demanded skills: {', '.join(top_skills)}\n"
        f"Role category distribution: {json.dumps(role_dist)}\n"
        f"Unique companies: {df['company'].nunique()}\n"
        f"Scraped dates: {sorted(df['scraped_date'].unique().tolist())}\n"
    )
    docs.append(Document(page_content=summary, metadata={"type": "summary", "season": season, "year": year}))
    print(f"  📄 Built {len(docs)} documents for {season} {year}")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════

import time  # Ensure time is imported at the top of your file if it isn't already

def _build_vector_store(docs: list, api_key: str = None) -> FAISS:
    # Fallback to env variable if api_key parameter isn't provided directly
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("RAG Vector Store initialization failed: 'GOOGLE_API_KEY' is missing or not set in environment.")

    # Separate summary docs (keep whole) from job docs (can split if huge)
    summary_docs = [d for d in docs if d.metadata.get('type') == 'summary']
    job_docs = [d for d in docs if d.metadata.get('type') != 'summary']

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_docs = summary_docs + splitter.split_documents(job_docs)

    print(f"  ⚡ Found {len(final_docs)} total chunks. Initializing rate-limited FAISS build...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=api_key
    )

    # ── RATE LIMIT BYPASS: Batching Document Ingestion ──
    # The free tier allows 100 embedding requests/min. We'll use small batches.
    BATCH_SIZE = 25 
    DELAY_SECONDS = 15  # Pause between batches to guarantee we stay under the 100/min limit

    # Initialize the FAISS vector store with the first batch
    first_batch = final_docs[:BATCH_SIZE]
    print(f"  📦 Processing batch 1/{((len(final_docs) - 1) // BATCH_SIZE) + 1} ({len(first_batch)} chunks)...")
    db = FAISS.from_documents(first_batch, embeddings)

    # Progressively add subsequent batches with a strict cooldown delay
    for i in range(BATCH_SIZE, len(final_docs), BATCH_SIZE):
        batch = final_docs[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = ((len(final_docs) - 1) // BATCH_SIZE) + 1
        
        print(f"  ⏳ Sleeping for {DELAY_SECONDS}s to protect API rate limit limits...")
        time.sleep(DELAY_SECONDS)
        
        print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        db.add_documents(batch)

    print("  ✅ Vector store successfully built without exhausting quota limits!")
    return db

# ═══════════════════════════════════════════════════════════════════════════════
#  RAG MAIN PIPELINE CREATOR
# ═══════════════════════════════════════════════════════════════════════════════

_init_errors = {}

def init_rag(season: str, year: int) -> bool:
    season = season.capitalize()
    key = f"{season}_{year}"
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")

    if not os.path.exists(parquet_path):
        _init_errors[key] = f"Parquet dataset file not found: {parquet_path}"
        return False

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _init_errors[key] = "Missing GOOGLE_API_KEY environment variable."
        return False

    try:
        docs = _load_parquet_docs(parquet_path, season, year)
        db = _build_vector_store(docs, api_key=api_key)
        retriever = db.as_retriever(search_kwargs={"k": 6})

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.3
        )

        # Build LCEL context chain
        def format_docs(documents):
            return "\n\n---\n\n".join(d.page_content for d in documents)

        context_chain = RunnableLambda(retriever) | RunnableLambda(format_docs)

        # Context-aware Prompt Setup
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Co-op Analytics Chatbot Assistant, an expert data concierge.\n"
                "Your objective is to answer questions about the scraped co-op and internship jobs for {season} {year}.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Base your answer strictly on the provided Context documents extracted from the dataset.\n"
                "2. If the user asks general or statistical metrics (e.g. total job count, top skills), look at the 'DATASET SUMMARY' document block provided inside the context.\n"
                "3. If details are absent, say: 'I cannot find that information in the dataset.' Do not hallucinate or make up metrics.\n"
                "4. Be structured, professional, and clear.\n\n"
                "Context Data:\n{context}"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Core chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {
                "context": context_chain,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "season": lambda x: season,
                "year": lambda x: str(year)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        _vector_stores[key] = db
        _rag_chains[key] = rag_chain
        _init_errors[key] = None
        print(f"  ✅ RAG Pipeline successfully initialized for {key}!")
        return True

    except Exception as e:
        import traceback
        err_msg = f"RAG initialisation error: {str(e)}"
        print(f"  ❌ {err_msg}")
        traceback.print_exc()
        _init_errors[key] = err_msg
        return False


# ── FastAPI Endpoints ─────────────────────────────────────────────────────────

chat_router = APIRouter(prefix="/api/chat", tags=["Chatbot"])

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    season: str = "Summer"
    year: int = 2026

class ChatResponse(BaseModel):
    answer: str
    season: str
    year: int
    ready: bool

@chat_router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    season = req.season.capitalize()
    key = f"{season}_{req.year}"

    # Auto-initialize on demand if not ready
    if key not in _rag_chains:
        success = await asyncio.get_event_loop().run_in_executor(None, init_rag, season, req.year)
        if not success:
            err_msg = _init_errors.get(key, "Unknown RAG initialization error.")
            return ChatResponse(
                answer=f"⚠️ The chatbot couldn't start: {err_msg}",
                season=season, year=req.year, ready=False
            )

    # Manage rolling in-memory history (last 10 interactions)
    hist_key = f"{req.session_id}_{key}"
    if hist_key not in _histories:
        _histories[hist_key] = deque(maxlen=10)
    history = _histories[hist_key]

    chain = _rag_chains[key]
    
    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: chain.invoke({"question": req.message, "chat_history": list(history)})
        )
    except Exception as e:
        answer = f"⚠️ An error occurred while executing the chain: {str(e)}"

    # Record context sequence history
    history.append(HumanMessage(content=req.message))
    history.append(AIMessage(content=answer))

    return ChatResponse(answer=answer, season=season, year=req.year,
                        ready=f"{season}_{req.year}" in _rag_chains)


@chat_router.get("/status/{season}/{year}")
async def chat_status(season: str, year: int):
    season = season.capitalize()
    key = f"{season}_{year}"
    return {
        "season": season,
        "year": year,
        "ready": key in _rag_chains,
        "error": _init_errors.get(key),   # None when healthy
    }


@chat_router.post("/init/{season}/{year}")
async def chat_init(season: str, year: int):
    season = season.capitalize()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, init_rag, season, year)
    return {{"season": season, "year": year, "initialized": ok}}


if __name__ == "__main__":
    import sys
    season, year = "Summer", 2026
    print(f"\n🤖 Co-operator RAG — {season} {year}\n" + "=" * 50)

    if not init_rag(season, year):
        print("Failed. Check GOOGLE_API_KEY and parquet file.")
        sys.exit(1)
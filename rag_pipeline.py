"""
rag_pipeline.py
───────────────
Production-grade, highly resilient RAG pipeline powered by Gemini 2.5 Flash-Lite.
Implements multi-tenant session tracking, strict thread-safe atomic locks, 
regex cache key normalization, an outbound API concurrency semaphore, and 
an exponential backoff retry loop to handle 429 quota limits gracefully.
"""

import os
import re
import json
import asyncio
import time
import threading
from collections import Counter, deque

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages  import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

import faiss
from sentence_transformers import SentenceTransformer

from google import genai as _genai
from google.genai import types as _genai_types

from fastapi           import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic          import BaseModel

load_dotenv(override=False)

DATA_DIR = os.getenv("DATA_DIR", ".")
GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"

print(f"  [RAG Engine] Active Text Model: {GEMINI_MODEL}")

# Lazy-loaded embedding storage parameters
_embed_model = None   
_embedding_dim = 384  

# ── THREADING & CONCURRENCY CONTROL STRUCTURES ────────────────────────────────
_model_init_lock = threading.Lock()   # Safeguards lazy model initialization
_throttle_lock   = threading.Lock()   # Guarantees atomic lookups for active chats
_gemini_semaphore = threading.Semaphore(2)  # Strict bottleneck: max 2 active LLM execution workers

def get_embedding_model():
    """Thread-safe lazy-initializer for the local SentenceTransformer weights."""
    global _embed_model
    if _embed_model is None:
        with _model_init_lock:
            # Double-checked locking pattern
            if _embed_model is None:
                print("  [RAG Engine] Lazy Loading Local Embedding Model (all-MiniLM-L6-v2)...")
                _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ── PERFORMANCE & MULTI-TENANT ISOLATED STORAGE ───────────────────────────────
_vector_indexes: dict = {}  # In-memory FAISS indices
_doc_lookups:    dict = {}  # Document fragments mapped by key
_summaries:      dict = {}  # Macro statistics summaries
_rag_chains:     dict = {}  # Executable LangChain flow graphs
_init_errors:    dict = {}  

# Session Storage Architecture: { key: { session_id: deque(history) } }
_session_histories: dict = {}  

# Global String Caches
_response_cache: dict = {}  
_active_chats:   dict = {}  # Tracking timestamps to block duplicate double-clicks


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDICATED BUILD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vector_store(parquet_path: str, season: str, year: int):
    key = f"{season}_{year}"
    index_disk_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
    docs_disk_path  = os.path.join(DATA_DIR, f"docs_{key}.json")
    sum_disk_path   = os.path.join(DATA_DIR, f"summary_{key}.json")

    # Fast-Path: Instantly pull compiled indices from local disk storage if found
    if os.path.exists(index_disk_path) and os.path.exists(docs_disk_path) and os.path.exists(sum_disk_path):
        print(f"  [FAISS Disk Cache] Loading index artifacts for {key}...")
        _vector_indexes[key] = faiss.read_index(index_disk_path)
        with open(docs_disk_path, "r", encoding="utf-8") as f:
            cached_docs = json.load(f)
            _doc_lookups[key] = [Document(page_content=d["p"], metadata=d["m"]) for d in cached_docs]
        with open(sum_disk_path, "r", encoding="utf-8") as f:
            _summaries[key] = json.load(f)["summary"]
        return

    print(f"  [⚠️ WARN] Index missing for {key}. Compiling from Parquet dataset...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet data file missing at: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(lambda x: list(x) if isinstance(x, (set, list)) else [])
    else:
        df["skills_found"] = [[] for _ in range(len(df))]

    docs = []
    for _, row in df.head(300).iterrows():
        skills = ", ".join(row["skills_found"]) if row["skills_found"] else "not specified"
        desc_raw = str(row.get("description", ""))[:300].replace("\n", " ")
        desc_snippet = desc_raw[:120] + "..." if len(desc_raw) > 120 else desc_raw
        
        text = (
            f"Job: {row.get('title', 'Unknown')} @ {row.get('company', 'Unknown')}\n"
            f"Loc: {row.get('location_city', 'Unknown')} | Remote: {row.get('is_remote', False)}\n"
            f"Skills: {skills}\n"
            f"Tasks: {desc_snippet}"
        )
        docs.append(Document(
            page_content=text,
            metadata={"title": str(row.get("title", "")).lower(), "company": str(row.get("company", "")).lower()}
        ))

    embed_engine = get_embedding_model()
    print(f"  [Embedding ENGINE] Encoding {len(docs)} jobs into vector space...")
    texts = [doc.page_content for doc in docs]
    embeddings = embed_engine.encode(texts, batch_size=32, show_progress_bar=False).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(_embedding_dim)
    index.add(embeddings)

    total = len(df)
    remote_pct = round(df["is_remote"].mean() * 100, 1) if total else 0
    top_skills = [s for s, _ in Counter([s for r in df["skills_found"] for s in r]).most_common(15)]
    summary_text = f"STATS SUMMARY: Unique jobs={total}, Remote={remote_pct}%\nTop Skills: {', '.join(top_skills)}\n"

    _vector_indexes[key] = index
    _doc_lookups[key]    = docs
    _summaries[key]      = summary_text

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        faiss.write_index(index, index_disk_path)
        with open(docs_disk_path, "w", encoding="utf-8") as f:
            json.dump([{"p": d.page_content, "m": d.metadata} for d in docs], f)
        with open(sum_disk_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text}, f)
        print(f"  [Disk Storage] Saved structural index cache snapshots for {key}")
    except Exception as e:
        print(f"  [Disk Storage Warn] Skipping file serialization: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SEARCH AND INFERENCE CHAINS WITH EXPONENTIAL BACKOFF RETRIES
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are Co-operator AI, an assistant for Canadian co-op job analytics.\n"
    "Context rules:\n"
    "- Give short, direct answers with crisp formatting.\n"
    "- For individual jobs, strictly cite company & title.\n"
    "- If query needs macro statistics or generic metrics, rely entirely on the STATS SUMMARY section.\n"
    "Data:\n{context}"
)

def _build_chain(season: str, year: int, api_key: str) -> RunnableLambda:
    client = _genai.Client(api_key=api_key)
    key = f"{season}_{year}"
    stats_keywords = {"top skills", "statistics", "most common", "distribution", "percent", "how many", "trend", "total"}

    def _retrieve(inputs: dict) -> dict:
        q_lower = inputs["question"].lower()
        index = _vector_indexes.get(key)
        all_docs = _doc_lookups.get(key, [])
        summary_payload = _summaries.get(key, "") if any(k in q_lower for k in stats_keywords) else ""

        matched_chunks = []
        if index is not None and all_docs:
            words = [w for w in q_lower.split() if len(w) > 3]
            company_filters = [d for d in all_docs if any(w in d.metadata["company"] for w in words)]
            
            embed_engine = get_embedding_model()
            if company_filters:
                sub_texts = [d.page_content for d in company_filters]
                sub_embs = embed_engine.encode(sub_texts, show_progress_bar=False).astype("float32")
                faiss.normalize_L2(sub_embs)
                q_emb = embed_engine.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                
                scores = np.dot(sub_embs, q_emb.T).flatten()
                for idx in np.argsort(-scores)[:3]:
                    if scores[idx] >= 0.20:
                        matched_chunks.append(company_filters[idx].page_content)
            else:
                q_emb = embed_engine.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                scores, indices = index.search(q_emb, k=3)
                for sim_score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and sim_score >= 0.20:
                        matched_chunks.append(all_docs[idx].page_content)

        if not matched_chunks and not summary_payload:
            matched_chunks = [d.page_content for d in all_docs[:2]]

        inputs["context"] = (f"{summary_payload}\n\n" if summary_payload else "") + "MATCHED JOBS:\n" + "\n---\n".join(matched_chunks)
        return inputs

    def _generate(inputs: dict) -> str:
        contents = []
        for msg in inputs.get("chat_history", []):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "model"
            contents.append(_genai_types.Content(role=role, parts=[_genai_types.Part.from_text(text=msg.content)]))

        contents.append(_genai_types.Content(role="user", parts=[_genai_types.Part.from_text(text=f"Context:\n{inputs['context']}\n\nQ: {inputs['question']}")]))
        cfg = _genai_types.GenerateContentConfig(system_instruction=_SYSTEM, temperature=0.15, max_output_tokens=250)
        
        # Concurrency Protection: Bottlenecks pipeline queries hitting upstream API servers
        with _gemini_semaphore:
            models_to_try = [GEMINI_MODEL, GEMINI_MODEL_FALLBACK]
            
            for model_target in models_to_try:
                max_retries = 3
                backoff_delay = 1.0  # Initial delay step in seconds
                
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model_target, 
                            contents=contents, 
                            config=cfg
                        )
                        return response.text
                    except Exception as e:
                        err_msg = str(e).lower()
                        
                        # Catch Quota, 429 Rate Limits, or Resource Exhaustion triggers explicitly
                        if "429" in err_msg or "exhausted" in err_msg or "rate_limit" in err_msg:
                            if attempt == max_retries - 1:
                                print(f"  🚨 [Quota Exhausted] Failed after {max_retries} bounds on model {model_target}.")
                                break  # Break loop step to exit and drop into fallback model chain
                            
                            print(f"  ⚠️ [429 Rate Limit] Hit on retry phase {attempt + 1}. Backing off for {backoff_delay}s...")
                            time.sleep(backoff_delay)
                            backoff_delay *= 2.0  # Exponential shift cascade: 1s -> 2s -> 4s
                        else:
                            raise e  # Fail instantly if it is an invalid API key, context syntax, or structural crash
                            
        raise HTTPException(
            status_code=429, 
            detail="The AI engine is experiencing a high volume of traffic. Please wait a moment before trying again."
        )

    return RunnableLambda(lambda inputs: _generate(_retrieve(inputs)))


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC LIFECYCLE INTERFACES WITH SESSION BUFFERING
# ═══════════════════════════════════════════════════════════════════════════════

def init_rag(season: str, year: int) -> bool:
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    key = f"{season}_{year}"
    path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")

    if key in _rag_chains:
        return True

    try:
        _build_vector_store(path, season, year)
        _rag_chains[key] = _build_chain(season, year, api_key)
        
        if key not in _session_histories:
            _session_histories[key] = {}
            
        _init_errors.pop(key, None)
        return True
    except Exception as e:
        _init_errors[key] = f"RAG Initialization Failure: {e}"
        print(f"  ❌ [CRITICAL] {_init_errors[key]}")
        return False


def ask(question: str, season: str, year: int, session_id: str = "default_user") -> str:
    key = f"{season}_{year}"
    
    # Normalized Cache Indexing: Scrub all symbols and minimize multi-whitespace groups
    normalized_q = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', question.strip().lower()))
    cache_key = f"{key}_{session_id}_{normalized_q}"
    
    if cache_key in _response_cache:
        print("  ⚡ [Cache Hit] Safely returning output context block from string cache map.")
        return _response_cache[cache_key]

    if key not in _rag_chains:
        index_disk_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
        if os.path.exists(index_disk_path):
            print(f"  [Hot Boot] Re-instantiating memory mapping from disk cache for {key}...")
            if not init_rag(season, year):
                return f"Error loading index: {_init_errors.get(key)}"
        else:
            return "Engine Notice: Data index for this season is currently building or not initialized yet."

    if session_id not in _session_histories[key]:
        _session_histories[key][session_id] = deque(maxlen=4)
        
    history_window = _session_histories[key][session_id]

    try:
        print(f"  🤖 [LLM Invoke] Processing query for session '{session_id}' via {GEMINI_MODEL}")
        answer = _rag_chains[key].invoke({"question": question, "chat_history": list(history_window)})
        
        history_window.append(HumanMessage(content=question))
        history_window.append(AIMessage(content=answer))
        
        _response_cache[cache_key] = answer
        return answer
    except Exception as e:
        print(f"  ❌ [CRITICAL PIPELINE EXCEPTION]: {str(e)}")
        return f"AI Generation processing constraint trace: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTER WITH ATOMIC LOCK THROTTLING
# ═══════════════════════════════════════════════════════════════════════════════

chat_router = APIRouter(prefix="/api/chat", tags=["chatbot"])

class ChatRequest(BaseModel):
    question:   str
    season:     str = "Summer"
    year:       int = 2026
    session_id: str = "default_user"  # Frontend tracking token/hash to ensure sandboxed histories

class ChatResponse(BaseModel):
    answer: str
    season: str
    year:   int
    ready:  bool

@chat_router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    season = req.season.capitalize()
    if season not in ["Summer", "Fall", "Winter"]:
        return JSONResponse(status_code=400, content={"error": "Invalid season configuration"})

    now = time.time()
    normalized_q = re.sub(r'\s+', ' ', req.question.strip().lower())
    throttle_key = f"{season}_{req.year}_{req.session_id}_{normalized_q}"
    
    # Thread-Safe Atomic Lock Evaluation Block
    with _throttle_lock:
        if throttle_key in _active_chats:
            last_time = _active_chats[throttle_key]
            if now - last_time < 2.0:  # Strict 2-second temporal click window boundary
                print("  🛑 [Throttled Lock] Concurrent duplicate query caught and dropped on backend!")
                raise HTTPException(status_code=429, detail="Duplicate client operation dropped.")
        _active_chats[throttle_key] = now

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask, req.question, season, req.year, req.session_id)
    return ChatResponse(answer=answer, season=season, year=req.year, ready=f"{season}_{req.year}" in _rag_chains)

@chat_router.get("/status/{season}/{year}")
async def chat_status(season: str, year: int):
    season = season.capitalize()
    key = f"{season}_{year}"
    return {"season": season, "year": year, "ready": key in _rag_chains, "model": GEMINI_MODEL, "error": _init_errors.get(key)}

@chat_router.post("/init/{season}/{year}")
async def chat_init(season: str, year: int):
    season = season.capitalize()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, init_rag, season, year)
    return {"season": season, "year": year, "initialized": ok, "model": GEMINI_MODEL, "error": _init_errors.get(f"{season}_{year}")}
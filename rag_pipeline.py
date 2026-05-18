"""
rag_pipeline.py
───────────────
Optimized High-Fidelity RAG pipeline powered by Gemini 2.5 Flash-Lite.
Leverages expanded context allowances, rich document vectorization, and structured
failsafes to deliver premium precision while preserving your API token runway.
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
_model_init_lock = threading.Lock()   
_throttle_lock   = threading.Lock()   
_gemini_semaphore = threading.Semaphore(4)  # Expanded concurrency headroom

def get_embedding_model():
    """Thread-safe lazy-initializer for local SentenceTransformer weights."""
    global _embed_model
    if _embed_model is None:
        with _model_init_lock:
            if _embed_model is None:
                print("  [RAG Engine] Lazy Loading Local Embedding Model (all-MiniLM-L6-v2)...")
                _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ── PERFORMANCE & MULTI-TENANT ISOLATED STORAGE ───────────────────────────────
_vector_indexes: dict = {}  
_doc_lookups:    dict = {}  
_summaries:      dict = {}  
_rag_chains:     dict = {}  
_init_errors:    dict = {}  

# Session Storage Architecture: { key: { session_id: deque(history) } }
_session_histories: dict = {}  

# Global String Caches
_response_cache: dict = {}  
_active_chats:   dict = {}  


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDICATED BUILD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vector_store(parquet_path: str, season: str, year: int):
    key = f"{season}_{year}"
    index_disk_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
    docs_disk_path  = os.path.join(DATA_DIR, f"docs_{key}.json")
    sum_disk_path   = os.path.join(DATA_DIR, f"summary_{key}.json")

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
    
    # DATA CORRECTION: Robustly handle structural parsing anomalies across the series
    def normalize_skills(val):
        if isinstance(val, (set, list, np.ndarray)):
            return [str(item).strip() for item in val if str(item).strip()]
        if isinstance(val, str) and val.strip():
            # If stored as a stringified JSON array or comma-separated tokens, unpack it
            if val.startswith('[') and val.endswith(']'):
                try: return [s.strip("'\" ") for s in json.loads(val)]
                except: pass
            return [s.strip() for s in val.split(',') if s.strip()]
        return []

    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(normalize_skills)
    else:
        df["skills_found"] = [[] for _ in range(len(df))]

    docs = []
    for _, row in df.head(600).iterrows():
        skills = ", ".join(row["skills_found"]) if row["skills_found"] else "not specified"
        desc_raw = str(row.get("description", "")).replace("\n", " ").strip()
        desc_snippet = desc_raw[:1500] + "..." if len(desc_raw) > 1500 else desc_raw
        
        text = (
            f"Job: {row.get('title', 'Unknown')} @ {row.get('company', 'Unknown')}\n"
            f"Loc: {row.get('location_city', 'Unknown')} | Remote: {row.get('is_remote', False)}\n"
            f"Skills Required: {skills}\n"
            f"Core Responsibilities: {desc_snippet}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "title": str(row.get('title', '')).lower().strip(), 
                "company": str(row.get('company', '')).lower().strip()
            }
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
    
    # DATA DEFENSE: Fall back to scanning the entire dataset if the first 600 rows are sparse
    all_skills = [s for r in df["skills_found"] for s in r if s]
    if not all_skills:
        # Fallback keyword extraction from job titles if skills column is completely blank
        all_skills = [w.capitalize() for t in df["title"].dropna() for w in str(t).split() if len(w) > 4]
        
    top_skills = [s for s, _ in Counter(all_skills).most_common(20)]
    
    # Safeguard against rendering an empty statistics block
    skills_string = ", ".join(top_skills) if top_skills else "General Technical/Communication Skillsets"
    summary_text = f"STATS SUMMARY: Unique jobs={total}, Remote={remote_pct}%\nTop Skills demanded this season: {skills_string}\n"

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


# ── SEARCH AND INFERENCE CHAINS WITH EXPONENTIAL BACKOFF RETRIES ─────────────────

_SYSTEM = (
    "You are Co-operator AI, an advanced analytical assistant for Canadian co-op job markets.\n"
    "Context Operational Rules:\n"
    "- Provide clear, descriptive, and comprehensive answers backed strictly by the provided text data.\n"
    "- If the user asks for macro insights, metrics, trends, or top skills, prioritize the values in the STATS SUMMARY section.\n"
    "- When listing target jobs, format them clearly as clean bullet points:\n"
    "  * [Job Title] at [Company Name] - [Location/Remote status]\n"
    "    -> Core Scope: Brief technical highlight of responsibilities or expected skills.\n"
    "- Always complete your formatting blocks fully. Never truncate mid-sentence or mid-bullet point.\n"
    "Data Context:\n{context}"
)

def _build_chain(season: str, year: int, api_key: str) -> RunnableLambda:
    client = _genai.Client(api_key=api_key)
    key = f"{season}_{year}"
    
    stats_keywords = {
        "top skills", "skills to learn", "statistics", "most common", "distribution", 
        "percent", "how many", "trend", "total", "demanded skills", "what skills"
    }

    def _retrieve(inputs: dict) -> dict:
        q_lower = inputs["question"].lower()
        index = _vector_indexes.get(key)
        all_docs = _doc_lookups.get(key, [])
        
        # Capture stats summary text context if any structural keywords hit
        summary_payload = _summaries.get(key, "") if any(k in q_lower for k in stats_keywords) else ""

        matched_chunks = []
        if index is not None and all_docs:
            ignore_words = {"in", "at", "to", "on", "by", "of", "an", "is", "me", "my", "do", "go", "no", "so", "or", "as", "if", "for", "with"}
            words = [w for w in q_lower.split() if len(w) >= 2 and w not in ignore_words]
            
            # Robust boundary checks for precise entity extraction (e.g., 'TD', 'RBC')
            company_filters = [
                d for d in all_docs 
                if any(re.search(rf'\b{re.escape(w)}\b', d.metadata["company"]) for w in words)
            ]
            
            embed_engine = get_embedding_model()
            if company_filters:
                sub_texts = [d.page_content for d in company_filters]
                sub_embs = embed_engine.encode(sub_texts, show_progress_bar=False).astype("float32")
                faiss.normalize_L2(sub_embs)
                q_emb = embed_engine.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                
                scores = np.dot(sub_embs, q_emb.T).flatten()
                # ROBUSTNESS UPGRADE: Pull up to 8 exact match company listings if available
                for idx in np.argsort(-scores)[:8]:
                    if scores[idx] >= 0.12:
                        matched_chunks.append(company_filters[idx].page_content)
            else:
                q_emb = embed_engine.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                
                # ROBUSTNESS UPGRADE: Expanded K depth from 6 to 12.
                # Gathers wide semantic coverage across your newly enriched descriptions.
                scores, indices = index.search(q_emb, k=12)
                for sim_score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and sim_score >= 0.18:
                        matched_chunks.append(all_docs[idx].page_content)

        if not matched_chunks and not summary_payload:
            matched_chunks = [d.page_content for d in all_docs[:3]]

        # Consolidate text data structure cleanly with absolute explicit tags
        context_str = ""
        if summary_payload:
            context_str += f"STATS SUMMARY:\n{summary_payload}\n\n"
        
        context_str += "MATCHED JOBS:\n" + "\n---\n".join(matched_chunks)
        inputs["context"] = context_str
        return inputs

    def _generate(inputs: dict) -> str:
        contents = []
        for msg in inputs.get("chat_history", []):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "model"
            contents.append(_genai_types.Content(role=role, parts=[_genai_types.Part.from_text(text=msg.content)]))

        contents.append(_genai_types.Content(role="user", parts=[_genai_types.Part.from_text(text=f"Context Documents:\n{inputs['context']}\n\nUser Question: {inputs['question']}")]))
        
        # ROBUSTNESS UPGRADE: Max output headroom scaled safely to 750 tokens.
        # Gives the model the room to construct beautiful, deeply rich insights without truncating.
        cfg = _genai_types.GenerateContentConfig(system_instruction=_SYSTEM, temperature=0.20, max_output_tokens=750)
        
        with _gemini_semaphore:
            models_to_try = [GEMINI_MODEL, GEMINI_MODEL_FALLBACK]
            for model_target in models_to_try:
                max_retries = 3
                backoff_delay = 1.0  
                
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model_target, contents=contents, config=cfg
                        )
                        return response.text
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "429" in err_msg or "exhausted" in err_msg or "rate_limit" in err_msg:
                            if attempt == max_retries - 1:
                                break  
                            time.sleep(backoff_delay)
                            backoff_delay *= 2.0  
                        else:
                            raise e  
                            
        raise HTTPException(status_code=429, detail="Upstream inference models are saturated. Retry your query shortly.")

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
    
    normalized_q = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', question.strip().lower()))
    cache_key = f"{key}_{session_id}_{normalized_q}"
    
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    if key not in _rag_chains:
        index_disk_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
        if os.path.exists(index_disk_path):
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
        
        if answer.strip().endswith("/") or answer.strip().endswith("-"):
            answer = answer.strip().rstrip("/-").strip() + "..."
            
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
    session_id: str = "default_user"  

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
    
    with _throttle_lock:
        if throttle_key in _active_chats:
            last_time = _active_chats[throttle_key]
            if now - last_time < 1.5:  
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
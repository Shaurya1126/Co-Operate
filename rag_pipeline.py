"""
rag_pipeline.py
───────────────
Ultra-optimized RAG pipeline powered by Gemini 2.5 Flash-Lite via the google.genai SDK.
Implements local semantic embeddings, FAISS Vector database indexing, disk persistence,
strict context text compression, conditional summary logic, active cache layering, 
and aggressive token minimization.
"""

import os
import json
import asyncio
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

from fastapi           import APIRouter
from fastapi.responses import JSONResponse
from pydantic          import BaseModel

load_dotenv(override=False)

DATA_DIR = os.getenv("DATA_DIR", ".")

# Model configurations - defaulting to highly cost-efficient 2.5-flash-lite
GEMINI_MODEL          = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash"

print(f"  [RAG Engine] Active Text Model: {GEMINI_MODEL}")

# Initialize local embedding model globally (Zero Gemini API costs for vectorization)
print("  [RAG Engine] Loading Local Embedding Model (all-MiniLM-L6-v2)...")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_embedding_dim = 384  # Dimensionality of all-MiniLM-L6-v2

# ── Performance & Optimization Memory Stores ──────────────────────────────────
_vector_indexes: dict = {}  # Global store for FAISS Inner Product indexes
_doc_lookups:    dict = {}  # Global mapping: {key: list_of_documents}
_summaries:      dict = {}  # Global aggregate analytical datasets
_rag_chains:     dict = {}  # Runnable LangChain pipelines
_histories:      dict = {}  # Restricted Chat history arrays
_init_errors:    dict = {}  # Pipeline initialization tracing
_response_cache: dict = {}  # High-impact global execution response cache


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT BUILDER, VECTOR INDEXER & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vector_store(parquet_path: str, season: str, year: int):
    key = f"{season}_{year}"
    index_disk_path = os.path.join(DATA_DIR, f"faiss_{key}.index")
    docs_disk_path  = os.path.join(DATA_DIR, f"docs_{key}.json")
    sum_disk_path   = os.path.join(DATA_DIR, f"summary_{key}.json")

    # Fast-Path: Load pre-built indices from persistent disk storage if available
    if os.path.exists(index_disk_path) and os.path.exists(docs_disk_path) and os.path.exists(sum_disk_path):
        print(f"  [FAISS Disk Cache] Loading index artifacts for {key}...")
        _vector_indexes[key] = faiss.read_index(index_disk_path)
        
        with open(docs_disk_path, "r", encoding="utf-8") as f:
            cached_docs = json.load(f)
            _doc_lookups[key] = [Document(page_content=d["p"], metadata=d["m"]) for d in cached_docs]
            
        with open(sum_disk_path, "r", encoding="utf-8") as f:
            _summaries[key] = json.load(f)["summary"]
        return

    # Compute-Path: Process Parquet data when cache is missing
    df = pd.read_parquet(parquet_path)

    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(
            lambda x: list(x) if isinstance(x, (set, list)) else []
        )
    else:
        df["skills_found"] = [[] for _ in range(len(df))]

    # Extract skills fallback block
    empty_pct = df["skills_found"].apply(lambda x: len(x) == 0).mean()
    if empty_pct > 0.8:
        print(f"  [Data Pipeline] {empty_pct:.0%} rows missing parsed items — re-extracting...")
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from report_generator import extract_skills as _extract, normalize_text as _norm
        full_text = (
            df.get("description", pd.Series("", index=df.index)).fillna("") + " " +
            df.get("highlights",  pd.Series("", index=df.index)).fillna("")
        )
        df["skills_found"] = full_text.apply(lambda t: _extract(_norm(t)))

    docs = []
    # Indexing top 300 data elements safely 
    for _, row in df.head(300).iterrows():
        skills = ", ".join(row["skills_found"]) if row["skills_found"] else "not specified"
        
        # Token Optimization: Extract compressed task data instead of massive open text blobs
        desc_raw = str(row.get("description", ""))[:300].replace("\n", " ")
        desc_snippet = desc_raw[:120] + "..." if len(desc_raw) > 120 else desc_raw
        
        # Super-compressed payload matching structure
        text = (
            f"Job: {row.get('title', 'Unknown')} @ {row.get('company', 'Unknown')}\n"
            f"Loc: {row.get('location_city', 'Unknown')}, {row.get('location_state', '')} | Remote: {row.get('is_remote', False)}\n"
            f"Skills: {skills}\n"
            f"Tasks: {desc_snippet}\n"
            f"Link: {row.get('apply_link', '')}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "title":   str(row.get("title",   "")).lower(),
                "company": str(row.get("company", "")).lower(),
                "season":  season.lower(),
                "year":    int(year),
            }
        ))

    # Generate Local Embeddings using CPU vectors
    print(f"  [Embedding] Encoding {len(docs)} objects locally via all-MiniLM-L6-v2...")
    texts = [doc.page_content for doc in docs]
    embeddings = _embed_model.encode(texts, batch_size=32, show_progress_bar=False).astype("float32")

    # High-Impact: Use Normalized Cosine Similarity (IndexFlatIP) instead of Euclidean L2
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(_embedding_dim)
    index.add(embeddings)

    # Compile Global Analytics Block separately (Keeps search context noise-free)
    dedup_cols = [c for c in ["title", "company", "apply_link"] if c in df.columns]
    unique_df  = df.drop_duplicates(subset=dedup_cols) if dedup_cols else df
    total      = len(unique_df)
    remote_pct = round(unique_df["is_remote"].mean() * 100, 1) if total else 0
    top_cities = unique_df["location_city"].value_counts().head(5).to_dict()
    all_skills = [s for r in unique_df["skills_found"] for s in r]
    top_skills = [s for s, _ in Counter(all_skills).most_common(20)]
    role_dist  = df["role_category"].value_counts().head(8).to_dict() if "role_category" in df.columns else {}

    summary_text = (
        f"STATS SUMMARY: Unique jobs={total}, Remote={remote_pct}%\n"
        f"Cities: {json.dumps(top_cities)}\n"
        f"Top Skills: {', '.join(top_skills)}\n"
        f"Roles: {json.dumps(role_dist)}\n"
    )

    # Commit to global state
    _vector_indexes[key] = index
    _doc_lookups[key]    = docs
    _summaries[key]      = summary_text

    # Persist objects immediately to disk for instantaneous future hot-starts
    try:
        faiss.write_index(index, index_disk_path)
        with open(docs_disk_path, "w", encoding="utf-8") as f:
            json.dump([{"p": d.page_content, "m": d.metadata} for d in docs], f)
        with open(sum_disk_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text}, f)
        print(f"  [Disk Storage] Vector database snapshot safely saved for key: {key}")
    except Exception as save_err:
        print(f"  [Disk Storage Warn] Could not cache artifact files to disk: {save_err}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HIGHLY CONDENSED SYSTEM INSTRUCTION Prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM = (
    "You are Co-operator AI, an assistant for Canadian co-op job analytics.\n"
    "Context rules:\n"
    "- Give short, direct answers with crisp formatting.\n"
    "- For individual jobs, strictly cite company & title.\n"
    "- If query needs macro statistics or generic metrics, rely entirely on the STATS SUMMARY section.\n"
    "- Reply from general training data for skill courses, interview tips, or general advice.\n\n"
    "Data:\n{context}"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SEARCH RETRIEVAL CHAIN WITH METADATA FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chain(season: str, year: int, api_key: str) -> RunnableLambda:
    client      = _genai.Client(api_key=api_key)
    system_text = _SYSTEM
    key         = f"{season}_{year}"

    # Token Optimization: Conditional keywords that signal macro summary insertion requirements
    stats_keywords = {"top skills", "statistics", "most common", "distribution", "percent", "how many", "trend", "total"}

    def _retrieve(inputs: dict) -> dict:
        q_lower = inputs["question"].lower()
        index   = _vector_indexes.get(key)
        all_docs = _doc_lookups.get(key, [])
        
        # Optimization 5: Conditional Summary Inclusion Evaluation
        needs_summary = any(k in q_lower for k in stats_keywords)
        summary_payload = _summaries.get(key, "") if needs_summary else ""

        matched_chunks = []

        if index is not None and all_docs:
            # Optimization 11: Metadata Filtering BEFORE Vector Exploration
            filtered_indices = range(len(all_docs))
            
            # Simple keyword extraction to check for exact company name filtering overrides
            words = [w for w in q_lower.split() if len(w) > 3]
            company_filters = [d for d in all_docs if any(w in d.metadata["company"] for w in words)]
            
            if company_filters:
                # If explicit tracking filters apply, execute vector alignment against that subset
                subset_texts = [d.page_content for d in company_filters]
                sub_embs = _embed_model.encode(subset_texts, show_progress_bar=False).astype("float32")
                faiss.normalize_L2(sub_embs)
                
                q_emb = _embed_model.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                
                scores = np.dot(sub_embs, q_emb.T).flatten()
                # Sort descending
                ranked_idx = np.argsort(-scores)
                
                # Optimization 4 & 13: Top 3 selection with matching score thresholds
                for idx in ranked_idx[:3]:
                    if scores[idx] >= 0.25: # Score threshold tuning
                        matched_chunks.append(company_filters[idx].page_content)
            else:
                # Execution Path: Global Dense Vector Store Sweep via Cosine Inner Product
                q_emb = _embed_model.encode([inputs["question"]]).astype("float32")
                faiss.normalize_L2(q_emb)
                
                # Optimization 4: Limit context search strictly to k=4 most relevant items
                scores, indices = index.search(q_emb, k=4)
                
                for sim_score, idx in zip(scores[0], indices[0]):
                    if idx != -1 and sim_score >= 0.25:  # Optimization 13: Retrieval threshold
                        matched_chunks.append(all_docs[idx].page_content)

        if not matched_chunks and not summary_payload:
            matched_chunks = [d.page_content for d in all_docs[:2]]

        # Optimization 14: Super-compressed payload layout injection
        inputs["context"] = (f"{summary_payload}\n\n" if summary_payload else "") + "MATCHED JOBS:\n" + "\n---\n".join(matched_chunks)
        return inputs

    def _generate(inputs: dict) -> str:
        contents = []
        
        # Optimization 8: Truncate message history length to a max threshold of 3 items
        for msg in inputs.get("chat_history", []):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "model"
            contents.append(
                _genai_types.Content(role=role, parts=[_genai_types.Part.from_text(text=msg.content)])
            )

        final_query = f"Context:\n{inputs['context']}\n\nQ: {inputs['question']}"
        contents.append(
            _genai_types.Content(role="user", parts=[_genai_types.Part.from_text(text=final_query)])
        )

        # Optimization 7: Reduce output generation capacity to 250 tokens
        cfg = _genai_types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=0.2,
            max_output_tokens=250,
        )

        models_to_try = [GEMINI_MODEL]
        if GEMINI_MODEL != GEMINI_MODEL_FALLBACK:
            models_to_try.append(GEMINI_MODEL_FALLBACK)

        last_err = None
        for model in models_to_try:
            try:
                response = client.models.generate_content(model=model, contents=contents, config=cfg)
                return response.text
            except Exception as e:
                last_err = e
                err_str = str(e)
                if any(c in err_str for c in ["403", "404", "invalid", "not found", "deprecated"]):
                    continue
                raise

        raise last_err

    def _chain(inputs: dict) -> str:
        return _generate(_retrieve(inputs))

    return RunnableLambda(_chain)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API WITH RESPONSE CACHING LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def init_rag(season: str, year: int) -> bool:
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    key     = f"{season}_{year}"
    path    = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")

    if not os.path.exists(path) and not os.path.exists(os.path.join(DATA_DIR, f"faiss_{key}.index")):
        _init_errors[key] = f"Missing data sources for execution matching path target: {path}"
        return False

    if not api_key:
        _init_errors[key] = "Missing critical authorization key: GOOGLE_API_KEY"
        return False

    try:
        _build_vector_store(path, season, year)
        _rag_chains[key] = _build_chain(season, year, api_key)
        # Optimization 8: Limit Chat Transcript Memory to tracking 4 segments total
        _histories[key]  = deque(maxlen=4)
        _init_errors.pop(key, None)
        return True
    except Exception as e:
        _init_errors[key] = f"RAG Fatal Initialization Tracer: {e}"
        return False


def ask(question: str, season: str, year: int) -> str:
    key = f"{season}_{year}"
    
    # Optimization 9: Active Response Cache Layer Validation
    cache_key = f"{key}_{question.strip().lower()}"
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    if key not in _rag_chains:
        if not init_rag(season, year):
            return f"Error Trace: {_init_errors.get(key, 'Initialization failure')}"

    try:
        answer = _rag_chains[key].invoke({
            "question":     question,
            "chat_history": list(_histories[key]),
        })
        
        # Populate history and update transaction cache
        _histories[key].append(HumanMessage(content=question))
        _histories[key].append(AIMessage(content=answer))
        _response_cache[cache_key] = answer
        return answer
    except Exception as e:
        return f"AI Service Execution Constraint Error: {str(e)[:150]}"


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
        return JSONResponse(status_code=400, content={"error": "Invalid season string structure"})
    
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
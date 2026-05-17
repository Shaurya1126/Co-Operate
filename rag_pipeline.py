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

import pandas as pd
from dotenv import load_dotenv

# ── LangChain v0.3 imports (all from *-core / *-community / *-google-genai) ──
from langchain_text_splitters          import RecursiveCharacterTextSplitter
from langchain_core.documents          import Document
from langchain_core.prompts            import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages           import HumanMessage, AIMessage
from langchain_core.output_parsers     import StrOutputParser
from langchain_core.runnables          import RunnableLambda
from langchain_community.vectorstores  import FAISS
from langchain_google_genai            import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi           import APIRouter
from fastapi.responses import JSONResponse
from pydantic          import BaseModel

print(f"DEBUG GOOGLE_API_KEY = '{os.getenv('GOOGLE_API_KEY')}'")
DATA_DIR       = os.getenv("DATA_DIR", ".")
GEMINI_MODEL   = "gemini-2.5-flash-lite"

# ── In-memory stores ──────────────────────────────────────────────────────────
_vector_stores: dict = {}
_rag_chains:    dict = {}
_histories:     dict = {}   # key → deque of LangChain message objects


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
        desc   = str(row.get("description", ""))[:600]
        text   = (
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
    dedup_cols  = [c for c in ["title", "company", "apply_link"] if c in df.columns]
    unique_df   = df.drop_duplicates(subset=dedup_cols) if dedup_cols else df
    total       = len(unique_df)
    remote_pct  = round(unique_df["is_remote"].mean() * 100, 1) if total else 0
    top_cities  = unique_df["location_city"].value_counts().head(5).to_dict()
    all_skills  = [s for row in unique_df["skills_found"] for s in row]
    top_skills  = [s for s, _ in Counter(all_skills).most_common(20)]
    role_dist  = (df["role_category"].value_counts().head(8).to_dict()
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
    docs.append(Document(page_content=summary,
                         metadata={"type": "summary", "season": season, "year": year}))

    print(f"  📄 Built {len(docs)} documents for {season} {year}")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vector_store(docs: list, api_key: str) -> FAISS:
    # Separate summary docs (keep whole) from job docs (can split if huge)
    summary_docs = [d for d in docs if d.metadata.get('type') == 'summary']
    job_docs     = [d for d in docs if d.metadata.get('type') != 'summary']

    # Only split job docs that exceed the chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", " "]
    )
    job_chunks = splitter.split_documents(job_docs)

    # Always keep summary docs intact (they contain the skill aggregates)
    all_chunks = summary_docs + job_chunks
    print(f"  🔢 {len(all_chunks)} chunks ({len(summary_docs)} summary + {len(job_chunks)} job) → embedding locally...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    vs = FAISS.from_documents(all_chunks, embeddings)
    print("  ✅ Vector store ready")
    return vs


# ═══════════════════════════════════════════════════════════════════════════════
#  LCEL CHAIN  (LangChain v0.3 — no ConversationalRetrievalChain, no Memory class)
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
    "- When asked about skills, ALWAYS look for the DATASET SUMMARY section in the context —\n"
    "  it lists the top 20 demanded skills. Use those numbers to answer. Never say skills are\n"
    "  not specified if a summary document is present in the context.\n"
    "- Individual job docs may say 'Skills Required: not specified' — ignore those and use\n"
    "  the summary aggregate instead when answering skill-related questions.\n"
    "- For questions about HOW TO LEARN a skill (tutorials, courses, resources, tips):\n"
    "  Answer freely using your general knowledge. Do NOT say you lack resources —\n"
    "  you are a career assistant and helping students learn is core to your role.\n"
    "- For questions about salaries, interview prep, resume tips, career paths:\n"
    "  Answer using your general knowledge, optionally grounding it in the job data.\n"
    "- Only say you cannot help if the question is completely unrelated to careers or jobs.\n\n"
    "Retrieved job context:\n{context}"
)


def _build_chain(vs: FAISS, season: str, year: int, api_key: str):
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=1024,
    )
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 30})

    system_text = _SYSTEM.replace("{season}", season).replace("{year}", str(year))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def retrieve_and_inject(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        # Always prepend the summary doc for skill/stats queries so the
        # model always sees the aggregated skill counts
        skill_keywords = ["skill", "skills", "require", "top", "most", "common",
                          "demand", "popular", "needed", "languages", "tools"]
        q_lower = inputs["question"].lower()
        if any(kw in q_lower for kw in skill_keywords):
            summary_results = vs.similarity_search(
                "dataset summary skills demanded", k=1,
                filter={"type": "summary"}
            )
            # Prepend summary so it appears first in context
            seen = {d.page_content for d in summary_results}
            docs = summary_results + [d for d in docs if d.page_content not in seen]
        inputs["context"] = "\n\n".join(d.page_content for d in docs)
        return inputs

    chain = RunnableLambda(retrieve_and_inject) | prompt | llm | StrOutputParser()
    return chain


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

_init_errors: dict = {}   # key → human-readable failure reason


def init_rag(season: str, year: int) -> bool:
    load_dotenv(override=False)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    key = f"{season}_{year}"
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")

    if not os.path.exists(parquet_path):
        msg = (f"No job data file found at '{parquet_path}'. "
               f"Please scrape {season} {year} jobs from the dashboard first.")
        print(f"  ⚠️  RAG: {msg}")
        _init_errors[key] = msg
        return False

    if not GOOGLE_API_KEY:
        msg = ("GOOGLE_API_KEY is not set in your environment / .env file. "
               "The chatbot requires a valid Gemini API key to function.")
        print(f"  ⚠️  RAG: {msg}")
        _init_errors[key] = msg
        return False

    try:
        print(f"  🤖 RAG: initialising chain for {key} (parquet={parquet_path}) …")
        docs = _load_parquet_docs(parquet_path, season, year)
        vs   = _build_vector_store(docs)
        _vector_stores[key] = vs
        _rag_chains[key]    = _build_chain(vs, season, year)
        _histories[key]     = deque(maxlen=12)   # 6 turns × 2 messages
        _init_errors.pop(key, None)              # clear any previous error
        print(f"  ✅ RAG chain ready for {key}")
        return True
    except Exception as e:
        import traceback
        msg = f"RAG initialisation error: {e}"
        print(f"  ❌ {msg}\n{traceback.format_exc()}")
        _init_errors[key] = msg
        return False
    


def ask(question: str, season: str, year: int) -> str:
    key = f"{season}_{year}"
    if key not in _rag_chains:
        ok = init_rag(season, year)
        if not ok:
            reason = _init_errors.get(key, "Unknown initialisation error.")
            return f"⚠️ The chatbot couldn't start: {reason}"

    chain   = _rag_chains[key]
    history = list(_histories[key])

    try:
        answer = chain.invoke({"question": question, "chat_history": history})
        _histories[key].append(HumanMessage(content=question))
        _histories[key].append(AIMessage(content=answer))
        return answer
    except Exception as e:
        return f"Error generating response: {e}"


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
    return ChatResponse(answer=answer, season=season, year=req.year,
                        ready=f"{season}_{req.year}" in _rag_chains)


@chat_router.get("/status/{season}/{year}")
async def chat_status(season: str, year: int):
    season = season.capitalize()
    key    = f"{season}_{year}"
    return {
        "season": season,
        "year":   year,
        "ready":  key in _rag_chains,
        "error":  _init_errors.get(key),   # None when healthy
    }


@chat_router.post("/init/{season}/{year}")
async def chat_init(season: str, year: int):
    season = season.capitalize()
    loop   = asyncio.get_event_loop()
    ok     = await loop.run_in_executor(None, init_rag, season, year)
    return {"season": season, "year": year, "initialized": ok}


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    season, year = "Summer", 2026
    print(f"\n🤖 Co-operator RAG — {season} {year}\n" + "=" * 50)

    if not init_rag(season, year):
        print("Failed. Check GOOGLE_API_KEY and parquet file.")
        sys.exit(1)

    for q in [
        "How many jobs are in the dataset?",
        "What are the top 5 most demanded technical skills?",
        "Are there remote software engineering positions?",
        "Which companies are hiring the most?",
    ]:
        print(f"\n❓ {q}\n💬 {ask(q, season, year)}")
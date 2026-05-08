"""
api.py
──────
FastAPI backend for the Co-op Readiness Report Generator.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000
"""

import os
import sys
import re
import uuid
import json
import shutil
import tempfile
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter
from datetime    import datetime
from pathlib     import Path

import requests as req_lib
import time as time_lib

from fastapi                 import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import FileResponse, StreamingResponse
from fastapi.staticfiles     import StaticFiles
from pydantic                import BaseModel
from dotenv import load_dotenv
import pytz
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", ".")

sys.path.insert(0, os.path.dirname(__file__))
from report_generator import (
    SKILLS, SOFT_SKILLS, DEGREE_ROLE_MAP, DEGREE_BASELINE_SKILLS,
    CANADIAN_UNIVERSITIES, AVAILABLE_DEGREES,
    API_KEY, NUM_PAGES,
    load_data, clean_dataframe, build_features, build_skill_freq,
    build_skill_trends, save_skill_charts, save_trend_chart,
    save_job_match_chart,
    score_student_against_all_jobs,
    extract_skills_from_resume, load_resume, extract_name_from_resume,
    generate_pdf_report,
    build_dataframe, should_refresh, write_lock, trim_old_data, build_rising_resources
)

# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Co-op Readiness API", version="1.0.0")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("praccy.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

_scrape_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMING SCRAPE
# ══════════════════════════════════════════════════════════════════════════════

def get_upcoming_seasons() -> list[dict]:
    """Returns the next 3 upcoming seasons with their correct years."""
    now   = datetime.now(pytz.timezone("America/New_York"))
    month = now.month
    year  = now.year

    # Define season start months
    season_order = [
        ("Summer", 5),   # May–Aug
        ("Fall",   9),   # Sep–Dec
        ("Winter", 1),   # Jan–Apr (of next year)
    ]

    # Find which season we're currently in or approaching
    upcoming = []
    for name, start_month in season_order:
        if name == "Winter":
            # Winter is always next calendar year if we're past Jan
            season_year = year + 1 if month >= 5 else year
        elif name == "Summer":
            season_year = year if month <= 8 else year + 1
        else:  # Fall
            season_year = year if month <= 12 else year + 1

        upcoming.append({"season": name, "year": season_year,
                         "label": f"{name} {season_year}"})

    # Sort by actual date so they appear in chronological order
    def season_date(s):
        m = {"Summer": 5, "Fall": 9, "Winter": 1}[s["season"]]
        y = s["year"] if s["season"] != "Winter" else s["year"]
        return (y, m)

    upcoming.sort(key=season_date)

    # Return only the 3 that are upcoming from today
    current_month_key = (year, month)
    future = [s for s in upcoming
              if season_date(s) >= current_month_key]

    # If fewer than 3 are in the future this year, roll into next year
    while len(future) < 3:
        last      = future[-1] if future else upcoming[-1]
        next_seasons = {"Summer": "Fall", "Fall": "Winter", "Winter": "Summer"}
        next_name  = next_seasons[last["season"]]
        next_year  = last["year"] + (1 if next_name == "Summer"
                                     and last["season"] == "Winter"
                                     else (1 if next_name == "Winter" else 0))
        future.append({"season": next_name, "year": next_year,
                        "label": f"{next_name} {next_year}"})

    return future[:3]

def build_search_query(season: str, year: int) -> str:
    return f"{season} Co-op OR Intern Canada {year}"

@app.get("/api/seasons")
def get_seasons():
    return {"seasons": get_upcoming_seasons()}

def fetch_jobs_streaming(query: str):
    """
    Generator — yields (page, count, all_jobs_so_far) after each page.
    Stops early if a page returns 0 results.
    """
    url     = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key":  API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    all_jobs = []

    for page in range(1, NUM_PAGES + 1):
        params = {
            "query":      query,
            "page":       str(page),
            "num_pages":  "1",
            "date_posted": "all",
        }
        jobs = []
        for attempt in range(3):
            try:
                response = req_lib.get(url, headers=headers,
                                       params=params, timeout=20)
                if response.status_code == 200:
                    jobs = response.json().get("data", [])
                else:
                    print(f"  API error {response.status_code}: {response.text}")
                break
            except req_lib.exceptions.Timeout:
                if attempt == 2:
                    break
                time_lib.sleep(2)
            except Exception:
                break

        all_jobs.extend(jobs)
        yield page, len(jobs), all_jobs

        # ── Stop early if page returned 0 results ─────────────────────────
        if len(jobs) == 0:
            print(f"  Page {page} returned 0 jobs — stopping early.")
            break

        time_lib.sleep(0.6)


def collect_streaming(season: str, year: int):
    """
    Generator that yields SSE-formatted strings while scraping.
    Used by /api/scrape-stream/{season}.
    """
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")
    legacy_path  = os.path.join(DATA_DIR, f"jobs_{season.lower()}.parquet")

    # Migrate legacy parquet (no year in name) to year-aware name if it exists and new one doesn't
    if not os.path.exists(parquet_path) and os.path.exists(legacy_path):
        import shutil as _shutil
        _shutil.copy2(legacy_path, parquet_path)
        write_lock(parquet_path)

    if not should_refresh(parquet_path):
        yield f"data: {json.dumps({'type':'cached','msg':'Data already collected today — loading from cache'})}\n\n"
        return
    query = build_search_query(season, year)
    yield f"data: {json.dumps({'type':'query','msg':f'Search query: {query}'})}\n\n"

    all_jobs = []
    for page, count, jobs_so_far in fetch_jobs_streaming(query):
        all_jobs = jobs_so_far
        yield f"data: {json.dumps({'type':'page','page':page,'total_pages':NUM_PAGES,'count':count,'total':len(jobs_so_far)})}\n\n"

    if not all_jobs:
        yield f"data: {json.dumps({'type':'error','msg':'No jobs returned. Check API key.'})}\n\n"
        return

    yield f"data: {json.dumps({'type':'building','msg':f'Building dataset from {len(all_jobs)} results...'})}\n\n"

    new_df = build_dataframe(all_jobs, season)

    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            today = datetime.now(pytz.timezone("America/New_York")).date().isoformat()
            if "scraped_date" in existing.columns:
                existing = existing[existing["scraped_date"] != today]
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
            subset=["title", "company", "scraped_date"], keep="last"
            ).reset_index(drop=True)
            new_df = combined
        except Exception:
            pass

    new_df.to_parquet(parquet_path, index=False)
    write_lock(parquet_path)
    trim_old_data(parquet_path, keep_days=90)

    # Invalidate memory cache so next request reloads fresh data
    _job_cache.pop(f"{season}_{year}", None)
    _job_cache.pop(season, None)  # also clear legacy key

    yield f"data: {json.dumps({'type':'done','total':len(new_df),'msg':f'Done. Saved {len(new_df)} jobs.'})}\n\n"


@app.get("/api/scrape-stream/{season}/{year}")
def scrape_stream(season: str, year: int):
    season = season.capitalize()
    if season not in ["Summer", "Fall", "Winter"]:
        raise HTTPException(400, "season must be Summer, Fall, or Winter")
    return StreamingResponse(
        collect_streaming(season, year),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/resources/{season}/{year}")
def get_resources(season: str, year: int):
    season = season.capitalize()
    cache = get_processed_jobs(season, year)
    trend_df = cache["trend_df"]
    rising = build_rising_resources(trend_df)
    return {"season": season, "available": len(rising) > 0, "resources": rising}

@app.get("/api/available-dates/{season}/{year}")
def get_available_dates(season: str, year: int):
    """Returns the list of scraped dates available in the dataset."""
    season = season.capitalize()
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")
    if not os.path.exists(parquet_path):
        return {"season": season, "dates": []}
    df = pd.read_parquet(parquet_path)
    if "scraped_date" not in df.columns:
        return {"season": season, "dates": []}
    dates = sorted(df["scraped_date"].dropna().unique().tolist(), reverse=True)
    return {"season": season, "dates": dates}

@app.get("/api/jobs/{season}/{year}/{date}")
def get_jobs_by_date(season: str, year: int, date: str):
    season = season.capitalize()
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet")
    if not os.path.exists(parquet_path):
        raise HTTPException(404, "No data for this season.")
    df = pd.read_parquet(parquet_path)
    df = df[df["scraped_date"] == date]
    if df.empty:
        raise HTTPException(404, f"No jobs found for date {date}.")
    df = clean_dataframe(df)
    df, _, _ = build_features(df)
    skill_df = build_skill_freq(df)
    if "posted_at" in df.columns:
        df["posted_at"] = df["posted_at"].dt.strftime("%Y-%m-%d").fillna("Unknown")
    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(
            lambda x: list(x) if isinstance(x, (set, list)) else []
        )
    jobs = df[["title", "company", "location_city", "is_remote",
               "apply_link", "role_category", "employment_type",
               "scraped_date", "skills_found"]].fillna("Unknown")
    skill_records = {
        "technical": skill_df[skill_df["type"] == "Technical"].head(20)[["skill","count","demand_pct"]].to_dict(orient="records"),
        "soft":      skill_df[skill_df["type"] == "Soft"].head(10)[["skill","count","demand_pct"]].to_dict(orient="records"),
    }
    return {
        "season": season, "date": date,
        "total": len(jobs),
        "jobs": jobs.to_dict(orient="records"),
        "skills": skill_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE HELPER
# ══════════════════════════════════════════════════════════════════════════════
_job_cache: dict = {}
_job_cache_times: dict = {}  # tracks when each key was cached

def get_processed_jobs(season: str, year: int = None) -> dict:
    cache_key    = f"{season}_{year}" if year else season
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_{year}.parquet") if year else os.path.join(DATA_DIR, f"jobs_{season.lower()}.parquet")

    print(f"DEBUG get_processed_jobs called with season='{season}' year='{year}' path='{parquet_path}'")

    # Invalidate cache if parquet was modified after we last cached it
    if cache_key in _job_cache:
        try:
            parquet_mtime = os.path.getmtime(parquet_path)
            cache_time = _job_cache_times.get(cache_key, 0)
            if parquet_mtime <= cache_time:
                return _job_cache[cache_key]  # still fresh
            else:
                print(f"  🔄 Parquet updated since last cache — reloading {cache_key}")
                del _job_cache[cache_key]
        except Exception:
            pass

    with _scrape_lock:
        if cache_key in _job_cache:
            return _job_cache[cache_key]

        if not os.path.exists(parquet_path):
            raise HTTPException(
                404,
                f"No data for {season} {year}. Scrape jobs first using the "
                f"/api/scrape-stream/{season}/{year} endpoint."
            )

        df = pd.read_parquet(parquet_path)
        df = clean_dataframe(df)
        df, tfidf, tfidf_matrix = build_features(df)
        skill_df = build_skill_freq(df)
        trend_df = build_skill_trends(season, year or 2026)

        save_skill_charts(skill_df, season)
        if not trend_df.empty:
            save_trend_chart(trend_df, season)

        _job_cache[cache_key] = {
            "df":           df,
            "skill_df":     skill_df,
            "trend_df":     trend_df,
            "tfidf":        tfidf,
            "tfidf_matrix": tfidf_matrix,
        }
        _job_cache_times[cache_key] = os.path.getmtime(parquet_path)
        return _job_cache[cache_key]



# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class GenerateReportRequest(BaseModel):
    season:           str
    year:             int = 2026
    name:             str
    university:       str
    degree:           str
    year_of_study:    int
    technical_skills: list[str]
    soft_skills:      list[str]


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/api/debug/{season}")
def debug_season(season: str):
    season = season.capitalize()
    parquet_path = os.path.join(DATA_DIR, f"jobs_{season.lower()}_2026.parquet")
    df = pd.read_parquet(parquet_path)
    return {
        "file": parquet_path,
        "season_values": df["season"].unique().tolist() if "season" in df.columns else "NO SEASON COLUMN",
        "row_count": len(df)
    }

@app.get("/api/meta")
def get_meta():
    return {
        "seasons":      ["Summer", "Fall", "Winter"],
        "degrees":      AVAILABLE_DEGREES,
        "universities": CANADIAN_UNIVERSITIES,
    }

@app.get("/api/jobs/{season}/{year}")
def get_jobs(season: str, year: int):
    season = season.capitalize()
    if season not in ["Summer", "Fall", "Winter"]:
        raise HTTPException(400, "Invalid season")
    cache = get_processed_jobs(season, year)
    df    = cache["df"].copy()

    if "posted_at" in df.columns:
        df["posted_at"] = df["posted_at"].dt.strftime("%Y-%m-%d").fillna("Unknown")

    # Convert skills_found list to JSON-serializable format
    if "skills_found" in df.columns:
        df["skills_found"] = df["skills_found"].apply(
            lambda x: list(x) if isinstance(x, (set, list)) else []
        )

    jobs = df[["title", "company", "location_city", "is_remote",
               "apply_link", "role_category", "employment_type",
               "scraped_date", "posted_at", "skills_found"]].fillna("Unknown")

    return {"season": season, "total": len(jobs),
            "jobs": jobs.to_dict(orient="records")}

@app.get("/api/fit-scores/{season}/{year}")
def get_fit_scores(season: str, year: int):
    """Returns fit score distribution across all jobs using average student profile."""
    season = season.capitalize()
    cache  = get_processed_jobs(season, year)
    df     = cache["df"]

    # Score all jobs using a generic baseline so the chart always has data
    from report_generator import DEGREE_BASELINE_SKILLS
    all_baseline_skills = list({s for skills in DEGREE_BASELINE_SKILLS.values() for s in skills})

    ranked = score_student_against_all_jobs(
        student_skills=all_baseline_skills,
        year_of_study=2,
        degree="Computer Science",
        df=df,
    )
    scores = ranked["fit_score"].tolist() if not ranked.empty else []
    return {"season": season, "scores": scores}


@app.get("/api/skills/{season}/{year}")
def get_skills(season: str, year: int):
    season   = season.capitalize()
    cache    = get_processed_jobs(season, year)
    skill_df = cache["skill_df"]
    tech     = skill_df[skill_df["type"] == "Technical"].head(20)
    soft     = skill_df[skill_df["type"] == "Soft"].head(10)
    return {
        "season":    season,
        "technical": tech[["skill", "count",
                            "demand_pct"]].to_dict(orient="records"),
        "soft":      soft[["skill", "count",
                            "demand_pct"]].to_dict(orient="records"),
    }


@app.get("/api/trends/{season}/{year}")
def get_trends(season: str, year: int):
    season   = season.capitalize()
    cache    = get_processed_jobs(season, year)
    trend_df = cache["trend_df"]
    if trend_df.empty:
        return {"season": season, "available": False, "trends": []}
    return {"season": season, "available": True,
            "trends": trend_df.head(20).to_dict(orient="records")}


@app.post("/api/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        text = load_resume(tmp.name)
        if not text.strip():
            raise HTTPException(422, "Could not extract text from resume.")
        name   = extract_name_from_resume(text)
        skills = extract_skills_from_resume(text)
        return {
            "detected_name":    name,
            "technical_skills": skills["technical"],
            "soft_skills":      skills["soft"],
            "total_found":      len(skills["all"]),
        }
    finally:
        os.unlink(tmp.name)

@app.post("/api/match-jobs")
def match_jobs(req: GenerateReportRequest):
    season     = req.season.capitalize()
    cache      = get_processed_jobs(season, req.year)
    all_skills = list(set(req.technical_skills + req.soft_skills))
    ranked     = score_student_against_all_jobs(
        student_skills=all_skills,
        year_of_study=req.year_of_study,
        degree=req.degree,
        df=cache["df"],
    )
    if ranked.empty:
        return {"matches": [], "total": 0}
    out = ranked.head(20).copy()
    for col in ["skills_required", "skills_matched",
                 "skills_missing", "skills_exposure"]:
        if col in out.columns:
            out[col] = out[col].apply(list)
    # Return all fit scores (not just top 20) so the fit distribution chart is accurate
    all_scores = ranked["fit_score"].tolist() if "fit_score" in ranked.columns else []
    return {"total": len(ranked), "matches": out.to_dict(orient="records"), "all_fit_scores": all_scores}


@app.post("/api/generate-report")
@app.post("/api/generate-report")
def generate_report(req: GenerateReportRequest):
    season     = req.season.capitalize()
    cache      = get_processed_jobs(season, req.year)
    all_skills = list(set(req.technical_skills + req.soft_skills))

    ranked = score_student_against_all_jobs(
        student_skills=all_skills,
        year_of_study=req.year_of_study,
        degree=req.degree,
        df=cache["df"],
    )

    safe_name = req.name.replace(" ", "_").lower()
    if not ranked.empty:
        save_job_match_chart(ranked, req.name, season)
    else:
        print(f"⚠️ No matches for {req.degree} Year {req.year_of_study} — generating report without matches")

    student = {
        "name":             req.name,
        "university":       req.university,
        "degree":           req.degree,
        "year_of_study":    req.year_of_study,
        "skills":           all_skills,
        "technical_skills": req.technical_skills,
        "soft_skills":      req.soft_skills,
        "resume_parsed":    True,
    }

    filename    = (f"coop_report_{safe_name}_{season.lower()}"
                   f"_{uuid.uuid4().hex[:6]}.pdf")
    output_path = str(OUTPUTS_DIR / filename)

    generate_pdf_report(
        student=student, season=season, ranked_jobs=ranked,
        skill_df=cache["skill_df"], trend_df=cache["trend_df"],
        df=cache["df"], output_path=output_path,
    )

    return {
        "success":       True,
        "filename":      filename,
        "download_url":  f"/outputs/{filename}",
        "total_matches": len(ranked),
        "top_match": {
            "title":     ranked.iloc[0]["job_title"],
            "company":   ranked.iloc[0]["company"],
            "fit_score": ranked.iloc[0]["fit_score"],
        } if len(ranked) else None,
    }


@app.get("/api/download/{filename}")
def download_report(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Report not found.")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=filename)

from scheduler import scheduler as job_scheduler
import atexit
atexit.register(lambda: job_scheduler.shutdown())
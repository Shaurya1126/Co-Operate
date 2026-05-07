# Co-Operate

> **Real-time co-op job market intelligence for Canadian university students.**  
> Scraped daily. Analyzed automatically. Personalized to your skills.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![Railway](https://img.shields.io/badge/Deployed_on-Railway-0B0D0E?style=flat-square&logo=railway&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Data Pipeline](#data-pipeline)
- [Report Engine](#report-engine)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Roadmap](#roadmap)

---

## Overview

Co-Operate is a full-stack job market intelligence platform built specifically for Canadian university co-op students. It scrapes real co-op job postings daily across all major seasons (Summer, Fall, Winter), analyzes skill demand trends over time, and generates personalized PDF reports that tell students exactly how competitive they are in the current market — and what to learn to close the gap.

Unlike static job boards, Co-Operate is a **living dataset**. Every day the platform automatically ingests new postings, computes skill frequency distributions, detects trending and declining skills, and surfaces curated learning resources for the skills that are rising fastest. The result is a market intelligence tool that gets smarter every day it runs.

---

## Features

### 🗂 Multi-Season Job Tracking
- Tracks **Summer, Fall, and Winter** co-op seasons simultaneously
- Season-aware data storage — each season's parquet file grows independently
- Dynamic season pills auto-populated from the API, no hardcoding required

### 📅 Historical Date Navigation
- Every job record is stamped with its `scraped_date`
- Users can navigate back to any previous scrape day using the date picker
- Charts and skill distributions update to reflect the selected snapshot in time

### 📊 Market Dashboard
- Live stats: jobs scraped, skills tracked, role categories, data freshness
- Top technical and soft skill bar charts with demand percentages
- Trending skills over the last 30 days (rising and falling)
- Curated learning resources for every rising skill, filterable by level (beginner / intermediate / advanced)

### 🔍 Insights Page
- **Role distribution** donut chart
- **Remote vs on-site** stacked bar chart broken down by role category
- **Skill demand** bars with filters for skill type (technical / soft) and count (top 10 / 15 / 20)
- **Top hiring companies** horizontal bar chart
- **Jobs posted over time** line chart, always rendered from the full unfiltered dataset
- **Fit score distribution** bar chart populated after report generation

### 💼 Job Browser
- Full paginated job listing with search by title or company
- Filter by role category
- Remote / on-site badge, apply link, role tag per listing
- Load more pagination (20 jobs per page)

### 📄 Personalized PDF Report Generation
- 4-step wizard: resume upload → personal details → skill review → report
- Resume parsing via AI — extracts technical and soft skills, detects name
- Fit scoring engine: matches user skills against every job in the dataset
- PDF includes: market overview, top skills analysis, trending skills, fit score breakdown, top job matches with apply links, and curated learning resources for rising skills the user doesn't yet have
- Downloadable PDF with live hyperlinks

### 🎓 Learning Resources Engine
- 100+ skills mapped to hand-curated learning resources
- Each resource tagged with a difficulty level (beginner / intermediate / advanced)
- Fuzzy + partial skill matching handles messy real-world job data (e.g. "Python programming" matches "Python")
- Alias resolution: "JS" → "JavaScript", "ML" → "Machine Learning", "postgres" → "PostgreSQL"
- Batch lookup function for processing multiple skills at once
- Surfaces resources both in the dashboard and in the PDF report

### ⚡ Real-Time Scrape Streaming
- Server-Sent Events (SSE) stream scrape progress live to the browser
- Per-page progress with job count, total so far, and percent complete
- Cache-aware: skips re-scraping if data is already fresh for the session
- Scrape logs rendered in a terminal-style panel with color-coded status lines

### 🕐 Automated Daily Scraping
- APScheduler runs the scraper automatically at **12:01 AM EST** every day
- Daylight saving handled automatically via `America/New_York` timezone
- New jobs are appended to the existing parquet file, preserving all historical data
- Data older than 90 days is automatically trimmed to keep file sizes manageable

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (index.html)                  │
│                                                             │
│  Dashboard │ Insights │ Browse Jobs │ Generate Report       │
│                                                             │
│  Season Selector ──── Date Picker (historical navigation)   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP / SSE
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (api.py)                   │
│                                                             │
│  /api/scrape-stream/{season}/{year}  ── SSE scrape stream   │
│  /api/jobs/{season}/{year}           ── job listings        │
│  /api/jobs/{season}/{year}/{date}    ── historical snapshot  │
│  /api/skills/{season}/{year}         ── skill frequencies   │
│  /api/trends/{season}/{year}         ── trending skills     │
│  /api/resources/{season}/{year}      ── learning resources  │
│  /api/available-dates/{season}/{year}── date picker data    │
│  /api/generate-report                ── PDF generation      │
│  /api/parse-resume                   ── resume AI parsing   │
│  /api/match-jobs                     ── fit scoring         │
│  /api/seasons                        ── dynamic seasons     │
│  /api/meta                           ── degrees/unis        │
└──────────┬────────────────────────────────┬─────────────────┘
           │                                │
           ▼                                ▼
┌──────────────────────┐       ┌────────────────────────────┐
│   Data Pipeline      │       │    Report Engine           │
│   (scraper)          │       │    (report_generator.py)   │
│                      │       │                            │
│  RapidAPI/JSearch    │       │  ReportLab PDF builder     │
│  BeautifulSoup       │       │  Skill matcher             │
│  Pandas cleaning     │       │  Fit scorer                │
│  Parquet storage     │       │  Learning resources        │
│  APScheduler cron    │       │  Resume parser (AI)        │
└──────────┬───────────┘       └────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Persistent Storage  │
│  /data/*.parquet     │
│  (Railway Volume)    │
└──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, Chart.js 4.4, Google Fonts (Syne + DM Sans) |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Data Processing | Pandas 2.0+, NumPy, PyArrow, scikit-learn |
| NLP / Parsing | spaCy, PyMuPDF, python-docx |
| PDF Generation | ReportLab |
| Scheduling | APScheduler |
| Storage | Apache Parquet (via PyArrow / fastparquet) |
| Deployment | Railway (persistent volume + environment variables) |
| Job Data Source | RapidAPI (JSearch) |
| Timezone Handling | pytz |
| Environment | python-dotenv |

---

## Project Structure

```
co-operate/
├── api.py                  # FastAPI application — all endpoints
├── report_generator.py     # PDF engine, skill matcher, fit scorer, learning resources
├── scheduler.py            # APScheduler daily scrape job
├── index.html              # Full frontend — single file SPA
├── requirements.txt        # Python dependencies
├── railway.json            # Railway deployment config
├── .env                    # Local environment variables (never committed)
├── .gitignore              # Excludes .env, parquet files, pycache
└── data/                   # Parquet files (Railway persistent volume in prod)
    ├── jobs_summer_2026.parquet
    ├── jobs_fall_2026.parquet
    └── jobs_winter_2027.parquet
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [RapidAPI](https://rapidapi.com) account with access to JSearch
- Node.js (only needed if you use the Railway CLI)

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/co-operate.git
cd co-operate
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install the spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

**5. Create your `.env` file**
```bash
cp .env.example .env
# Then edit .env and add your RapidAPI key
```

**6. Start the API**
```bash
uvicorn api:app --reload
```

**7. Open the frontend**

Open `index.html` in your browser directly, or visit `http://localhost:8000` if you're serving it from FastAPI.

---

## API Reference

### Scraping

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/scrape-stream/{season}/{year}` | SSE stream — scrapes jobs and streams progress |
| GET | `/api/seasons` | Returns all configured seasons with labels |
| GET | `/api/meta` | Returns degree and university lists for the report form |

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/jobs/{season}/{year}` | All jobs for a season (latest scrape) |
| GET | `/api/jobs/{season}/{year}/{date}` | Jobs for a specific historical scrape date |
| GET | `/api/available-dates/{season}/{year}` | All scrape dates available in the dataset |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/skills/{season}/{year}` | Skill frequency distribution (technical + soft) |
| GET | `/api/trends/{season}/{year}` | Rising and falling skills vs 30 days ago |
| GET | `/api/resources/{season}/{year}` | Learning resources for rising skills |

### Report

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/parse-resume` | Upload resume PDF/DOCX, returns extracted skills |
| POST | `/api/match-jobs` | Returns fit-scored job matches for a skill profile |
| POST | `/api/generate-report` | Generates and returns a downloadable PDF report |

---

## Data Pipeline

### Scrape Flow

```
User visits site
      │
      ▼
scrape-stream SSE endpoint called
      │
      ├─ Check cache: is parquet fresh? ──── YES ──► stream "cached" event, done
      │
      NO
      │
      ▼
Query JSearch API page by page
      │
      ├─ Stream "page" event per page (job count, progress %)
      │
      ▼
Clean & deduplicate raw job data
      │
      ▼
Extract skills using spaCy + SKILLS list matching
      │
      ▼
Append new rows to parquet file (preserve historical dates)
      │
      ▼
Trim rows older than 90 days
      │
      ▼
Stream "done" event ──► frontend renders dashboard
```

### Skill Detection

Skills are detected by matching job description text against a master `SKILLS` list of 150+ terms covering:

- Programming languages (Python, SQL, Java, Go, Rust, etc.)
- Web frameworks (React, Django, FastAPI, Node.js, etc.)
- Data science & ML (scikit-learn, TensorFlow, PyTorch, etc.)
- Data engineering (Spark, Kafka, Airflow, dbt, Snowflake, etc.)
- Cloud & DevOps (AWS, GCP, Azure, Docker, Kubernetes, etc.)
- Databases (PostgreSQL, MongoDB, Redis, Elasticsearch, etc.)
- Soft skills (Communication, Leadership, Problem Solving, etc.)
- Domain skills (Financial Modeling, Bioinformatics, IoT, etc.)

### Trend Detection

Trends are computed by comparing the current skill frequency distribution against the distribution from 30 days ago:

```
change = current_demand_pct - historical_demand_pct
```

Skills with `change > 0` are rising. Skills with `change < 0` are falling. Only skills with at least 2 data points (2+ scrape days) produce trend data.

---

## Report Engine

### Fit Scoring

Each job is scored against the user's skill profile using a weighted overlap formula:

```
fit_score = (matched_skills / required_skills) × 100
```

Jobs are then bucketed into tiers:

| Score | Tier |
|---|---|
| 85–100% | Excellent Match |
| 70–84% | Strong Match |
| 50–69% | Good Match |
| 30–49% | Partial Match |
| < 30% | Reach |

### Learning Resources

The resources engine maps 150+ skills to hand-curated learning links. It uses a three-stage lookup:

1. **Alias resolution** — "JS" → "JavaScript", "ML" → "Machine Learning", "postgres" → "PostgreSQL"
2. **Exact match** — direct case-insensitive key lookup
3. **Fuzzy match** — substring containment with a minimum 4-character guard to prevent false positives (e.g. "C" matching "Communication")

Resources are tagged with difficulty levels and surfaced for every rising skill the user doesn't already have in their profile.

### PDF Structure

Generated reports include:

1. Cover page with student name, university, degree, and season
2. Market overview (total jobs, top skills, role distribution)
3. Trending skills (rising and falling with percentage change)
4. Learning resources for rising skills (with clickable hyperlinks)
5. Personalized fit score summary
6. Top 10 job matches with company, location, fit score, matched skills, missing skills, and apply link

---

## Deployment

Co-Operate is deployed on [Railway](https://railway.com).

### One-time setup

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "initial commit"
git push origin main
```

**2. Create Railway project**
- Go to [railway.com](https://railway.com)
- New Project → Deploy from GitHub repo
- Select your repository

**3. Add persistent volume**
- New → Volume
- Attach to your service
- Mount path: `/data`

**4. Set environment variables**
- `RAPIDAPI_KEY` = your RapidAPI key
- `DATA_DIR` = `/data`

**5. Generate domain**
- Settings → Networking → Generate Domain

### Automatic daily scraping

The scheduler (`scheduler.py`) is imported by `api.py` on startup and runs the scraper automatically at **12:01 AM EST** every day via APScheduler. No external cron service required — it runs inside the same process as the API.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `RAPIDAPI_KEY` | Yes | Your RapidAPI key for JSearch |
| `DATA_DIR` | No | Directory for parquet files. Defaults to `.` locally, set to `/data` on Railway |

---

## Roadmap

- [ ] Email alerts when a new high-fit job is posted for a saved skill profile
- [ ] User accounts — save your profile and get a personalized feed
- [ ] Multi-country support (US co-op programs)
- [ ] Salary range extraction and visualization
- [ ] Company profile pages with historical hiring patterns
- [ ] Resume gap analysis — auto-detect which skills to prioritize based on your target roles
- [ ] Mobile app (React Native)
- [ ] Slack / Discord bot for daily job digest

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built for Canadian co-op students, by a Canadian co-op student.
</div>

"""
collect_jobs.py
───────────────
Script 1 — Data Collection
Scrapes co-op job postings via the JSearch API and saves them to a
season-specific parquet file that report_generator.py reads.

Usage:
    python collect_jobs.py
"""

import requests
import pandas as pd
import time
import datetime
import sys
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY     = "ee4e750272msha602a6de618b042p1dc00ajsn1088c183f321"
NUM_PAGES   = 13        # ~10 jobs per page → ~130 jobs total
OUTPUT_DIR  = "."       # folder where parquet files are saved


# ── SEASON SELECTION ──────────────────────────────────────────────────────────

def choose_season() -> str:
    """Prompt the user to select a co-op season."""
    seasons = {
        "1": "Summer",
        "2": "Fall",
        "3": "Winter",
    }

    print("\n" + "=" * 55)
    print("   Co-op Job Market — Data Collection")
    print("=" * 55)
    print("\nWhich co-op season would you like to scrape?\n")
    print("  1 — Summer  (May – August)")
    print("  2 — Fall    (September – December)")
    print("  3 — Winter  (January – April)")
    print()

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in seasons:
            season = seasons[choice]
            print(f"\n✅ Season selected: {season}\n")
            return season
        print("  ⚠️  Invalid choice — please enter 1, 2, or 3.")


def build_search_query(season: str) -> str:
    """Build a targeted API search query for the chosen season."""
    year = datetime.date.today().year

    # Determine the next occurrence of the season for the query year
    if season == "Summer":
        query_year = year if datetime.date.today().month <= 8 else year + 1
    elif season == "Fall":
        query_year = year if datetime.date.today().month <= 12 else year + 1
    else:  # Winter
        query_year = year + 1 if datetime.date.today().month >= 9 else year

    query = f"{season} Co-op OR Intern Canada {query_year}"
    print(f"🔍 Search query: \"{query}\"")
    return query


# ── API FETCH ─────────────────────────────────────────────────────────────────

def fetch_jobs(query: str, num_pages: int) -> list[dict]:
    """Fetch job postings from JSearch API across multiple pages."""
    url     = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key":  API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    all_jobs = []

    for page in range(1, num_pages + 1):
        print(f"  Fetching page {page}/{num_pages}...", end=" ")
        params = {
            "query":      query,
            "page":       str(page),
            "num_pages":  "1",
            "date_posted": "all",
        }

        try:
            response = requests.get(url, headers=headers,
                                    params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network error on page {page}: {e}")
            break

        if response.status_code != 200:
            print(f"\n  ⚠️  Error {response.status_code}: {response.text[:200]}")
            break

        data = response.json()
        jobs = data.get("data", [])
        print(f"got {len(jobs)} jobs")
        all_jobs.extend(jobs)

        time.sleep(0.6)   # polite rate limiting

    return all_jobs


# ── DATAFRAME BUILDER ─────────────────────────────────────────────────────────

def build_dataframe(jobs: list[dict], season: str) -> pd.DataFrame:
    """Parse raw API results into a clean DataFrame."""
    records = []
    for job in jobs:
        records.append({
            "title":           job.get("job_title"),
            "company":         job.get("employer_name"),
            "location_city":   job.get("job_city"),
            "location_state":  job.get("job_state"),
            "is_remote":       job.get("job_is_remote"),
            "employment_type": job.get("job_employment_type"),
            "description":     job.get("job_description"),
            "apply_link":      job.get("job_apply_link"),
            "source":          job.get("job_publisher"),
            "posted_at":       job.get("job_posted_at_datetime_utc"),
            "highlights":      str(job.get("job_highlights", {})),
            "season":          season,          # tag with season
        })

    df = pd.DataFrame(records)

    # ── Type cleanup ──────────────────────────────────────────────────────────
    df["posted_at"]  = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)
    df["is_remote"]  = df["is_remote"].astype(bool)

    # ── Fill nulls ────────────────────────────────────────────────────────────
    df["location_city"]  = df["location_city"].fillna("Unknown")
    df["location_state"] = df["location_state"].fillna("Unknown")
    df["employment_type"] = df["is_remote"].apply(
        lambda x: "Remote" if x else "In person"
    )

    # ── Deduplicate ───────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(
        subset=["title", "company"], keep="first"
    ).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  🧹 Removed {removed} duplicate postings")

    return df


# ── MERGE WITH EXISTING DATA ──────────────────────────────────────────────────

def merge_with_existing(new_df: pd.DataFrame,
                         output_path: str) -> pd.DataFrame:
    """
    If a parquet already exists for this season, merge new results with
    existing ones and deduplicate — so the dataset grows over time.
    """
    if not os.path.exists(output_path):
        return new_df

    print(f"  📂 Existing file found — merging with previous data...")
    try:
        existing_df = pd.read_parquet(output_path)
        combined    = pd.concat([existing_df, new_df], ignore_index=True)
        combined    = combined.drop_duplicates(
            subset=["title", "company"], keep="last"
        ).reset_index(drop=True)
        print(f"  📊 Previous: {len(existing_df)} | "
              f"New: {len(new_df)} | "
              f"After merge: {len(combined)}")
        return combined
    except Exception as e:
        print(f"  ⚠️  Could not read existing file ({e}), "
              f"starting fresh.")
        return new_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    season      = choose_season()
    query       = build_search_query(season)
    output_file = os.path.join(OUTPUT_DIR,
                               f"jobs_{season.lower()}.parquet")

    print(f"\n📡 Fetching jobs — this may take ~{NUM_PAGES * 0.6:.0f} seconds...\n")
    jobs = fetch_jobs(query, NUM_PAGES)

    if not jobs:
        print("\n❌ No jobs returned. Check your API key or network connection.")
        sys.exit(1)

    print(f"\n📦 Building dataframe from {len(jobs)} raw results...")
    new_df = build_dataframe(jobs, season)

    # Merge with existing season data if present
    final_df = merge_with_existing(new_df, output_file)

    # Save
    final_df.to_parquet(output_file, index=False)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Saved {len(final_df)} jobs → '{output_file}'")
    print(f"{'=' * 55}")
    print(f"\n  Season:   {season}")
    print(f"  Jobs:     {len(final_df)}")
    print(f"  Columns:  {list(final_df.columns)}")
    print(f"\n  ▶  Next step: run  python report_generator.py  "
          f"and select '{season}' when prompted.\n")

    # Preview
    print(final_df[["title", "company", "location_city",
                     "employment_type", "posted_at"]].head(5).to_string())
    print()


if __name__ == "__main__":
    main()

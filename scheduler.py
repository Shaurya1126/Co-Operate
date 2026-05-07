from apscheduler.schedulers.background import BackgroundScheduler
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

SEASONS = [
    ("Summer", 2026),
    ("Fall", 2026),
    ("Winter", 2027),
]

def daily_scrape():
    print("Running daily scrape...")
    for season, year in SEASONS:
        try:
            # Use regular get endpoint to trigger scrape and cache
            r = requests.get(f"{API_URL}/api/scrape-stream/{season}/{year}", timeout=300, stream=True)
            for line in r.iter_lines():
                print(line)
        except Exception as e:
            print(f"Scrape failed for {season} {year}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(daily_scrape, 'cron', hour=5, minute=1, timezone='America/New_York')
scheduler.start()
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
import requests
import os
import pytz

API_URL = os.getenv("API_URL", "http://localhost:8000")

SEASONS = [
    ("Summer", 2026),
    ("Fall",   2026),
    ("Winter", 2027),
]

def daily_scrape():
    print("🕐 Daily scrape starting...")
    for season, year in SEASONS:
        success = False
        for attempt in range(1, 4):  # retry up to 3 times
            try:
                print(f"  Scraping {season} {year} (attempt {attempt})...")
                r = requests.get(
                    f"{API_URL}/api/scrape-stream/{season}/{year}",
                    timeout=600,  # 10 min timeout — scraping takes time
                    stream=True
                )
                for line in r.iter_lines():
                    if line:
                        print(line)
                print(f"  ✅ {season} {year} done.")
                success = True
                break
            except Exception as e:
                print(f"  ⚠️  Attempt {attempt} failed for {season} {year}: {e}")
                if attempt < 3:
                    import time
                    time.sleep(30)  # wait 30s before retry
        if not success:
            print(f"  ❌ All attempts failed for {season} {year}")

jobstores = {'default': MemoryJobStore()}
scheduler = BackgroundScheduler(jobstores=jobstores, timezone=pytz.timezone("America/New_York"))
scheduler.add_job(
    daily_scrape,
    'cron',
    hour=0,
    minute=1,
    timezone='America/New_York',
    misfire_grace_time=3600,  # if job missed by up to 1 hour, still run it
    coalesce=True,            # if multiple missed, only run once
    max_instances=1           # never run more than one scrape at a time
)
scheduler.start()
print("✅ Scheduler started — daily scrape at 12:01 AM EST")
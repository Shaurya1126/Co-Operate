import pandas as pd
from datetime import date, timedelta

files = [
    "jobs_summer_2026.parquet",
    "jobs_fall_2026.parquet", 
    "jobs_winter_2027.parquet"
]

for filename in files:
    try:
        df = pd.read_parquet(filename)
        today = date.today().isoformat()
        
        # Add fake data for past 3 days
        fakes = []
        for i in range(1, 4):
            past_date = (date.today() - timedelta(days=i)).isoformat()
            fake = df[df["scraped_date"] == today].copy()
            if fake.empty:
                fake = df.copy()
            fake["scraped_date"] = past_date
            fakes.append(fake)
        
        combined = pd.concat([df] + fakes, ignore_index=True)
        combined.to_parquet(filename, index=False)
        print(f"{filename}: dates now = {sorted(combined['scraped_date'].unique())}")
    except FileNotFoundError:
        print(f"{filename} not found, skipping")
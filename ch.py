import pandas as pd
df = pd.read_parquet("jobs_summer_2026.parquet")
print(df.columns.tolist())
print(df["scraped_date"].unique() if "scraped_date" in df.columns else "NO scraped_date column")
print(len(df))
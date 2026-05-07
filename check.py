import pandas as pd
df = pd.read_parquet("jobs_fall_2026.parquet")
print(df["scraped_date"].unique())
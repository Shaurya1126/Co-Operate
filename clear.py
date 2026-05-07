# Run this once in a Python shell or add temporarily to main()
import os
for season in ["Summer", "Fall", "Winter"]:
    lock = f"jobs_{season.lower()}.lock"
    if os.path.exists(lock):
        os.remove(lock)
        print(f"Cleared lock for {season}")
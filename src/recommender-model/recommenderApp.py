import os
import pandas as pd
import subprocess
import shutil
from datetime import datetime

# Run query.py at the beginning
print("Running query.py...")
query_result = subprocess.run(['python', 'query.py'], capture_output=True, text=True)

# Check if the query script ran successfully
if query_result.returncode != 0:
    print("Query script failed:", query_result.stderr)
    exit()

print("Query script completed successfully.")

# Check if train.csv exists in the current directory
if not os.path.isfile('app/train.csv'):
    print("train.csv does not exist in the current directory.")
    exit()

# Load the data from train.csv
train_data = pd.read_csv('app/train.csv')

# Check the number of rows in the data
if len(train_data) <= 20:
    print("train.csv has 20 or fewer rows. No action will be taken.")
    exit()

# Run dockerRetrain.py
print("Running dockerRetrain.py...")
retrain_result = subprocess.run(['python', 'dockerRetrain.py'], capture_output=True, text=True)

# Check if the retrain script ran successfully
if retrain_result.returncode != 0:
    print("Retrain script failed:", retrain_result.stderr)
    exit()

print("Retrain script completed successfully.")

# Rename train.csv to the current date and move it to /app/archive
current_date = datetime.now().strftime("%Y-%m-%d")
archive_path = f'app/archive/train_{current_date}.csv'
shutil.move('app/train.csv', archive_path)
print(f"train.csv has been renamed to {archive_path} and moved to app/archive.")

# Rename retrain.csv to train.csv
shutil.move('app/retrain.csv', 'app/train.csv')
print("retrain.csv has been renamed to train.csv.")

print("Process completed successfully.")

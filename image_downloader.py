import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys
import time
import random
from urllib.request import Request, urlopen
import urllib.error

# Parse arguments
parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')
parser.add_argument('type', type=str, help='train, validate, or test')
args = parser.parse_args()

# Read data
df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# Progress bar
pbar = tqdm(total=len(df))

# Create image folder if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

def download_image(image_url, file_path):
    for attempt in range(5):
        try:
            req = Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
            return True  # Download successful
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 2 ** attempt
                print(f"429 Too Many Requests. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            elif e.code == 404:
                print(f"404 Not Found: Skipping {image_url}")
                return False  # Skip
            else:
                print(f"HTTP Error {e.code} for {image_url}: {e.reason}")
                return False
        except Exception as e:
            print(f"Unexpected error for {image_url}: {e}")
            return False
    return False  # Gave up after retries

# Main download loop
for index, row in df.iterrows():
    if row["hasImage"] == True and row["image_url"] not in ["", "nan"]:
        image_url = row["image_url"]
        file_path = os.path.join("images", f"{row['id']}.jpg")
        download_image(image_url, file_path)
        #time.sleep(random.uniform(1, 3))
    pbar.update(1)

pbar.close()
print("done")

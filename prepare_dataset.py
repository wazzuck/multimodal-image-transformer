"""
Downloads and prepares the Flickr30k dataset.

Checks if the dataset exists, otherwise downloads, extracts,
and converts captions to the required JSON format.
"""

import os
import requests
import zipfile
import shutil
import csv
import json
from pathlib import Path
from tqdm import tqdm

# Determine the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Configuration based on script location
DATA_DIR = SCRIPT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
CAPTIONS_JSON_PATH = DATA_DIR / "captions.json"
CAPTIONS_CSV_PATH = DATA_DIR / "captions.csv" # Temporary path after extraction (relative to DATA_DIR)
DOWNLOAD_DIR = SCRIPT_DIR / "temp_download" # Temporary directory for downloads (relative to script dir)

# URLs for Flickr30k parts from awsaf49/flickr-dataset release
FLICKR30K_URLS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]
CONCATENATED_ZIP_PATH = DOWNLOAD_DIR / "flickr30k.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "flickr30k_extracted" # Temp extraction dir

# --- Helper Functions ---

def download_file(url, dest_path):
    """Downloads a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except IOError as e:
        print(f"Error writing file {dest_path}: {e}")
        return False

def check_dataset_exists():
    """Checks if the dataset seems to be downloaded and prepared."""
    if not IMAGE_DIR.is_dir() or not CAPTIONS_JSON_PATH.is_file():
        return False
    # Basic check: does image dir have files?
    try:
        if not any(IMAGE_DIR.iterdir()):
            return False
        return True
    except StopIteration: # Handle empty directory case
         return False
    except OSError as e:
         print(f"Error checking directory {IMAGE_DIR}: {e}")
         return False

def convert_csv_to_json(csv_path, json_path):
    """
    Converts the Flickr30k results.csv to the target captions.json format.
    Assumes CSV format: image_name| comment_number| comment
    Output JSON format: { "image_filename": ["caption1", "caption2", ...] }
    """
    print(f"Converting {csv_path} to {json_path}...")
    captions = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Handle potential variations in header or lack thereof
            # Sniffing might be more robust but let's try common structure first
            reader = csv.reader(csvfile, delimiter='|')
            header = next(reader) # Skip header if present

            # Try to infer column indices dynamically if possible, otherwise assume defaults
            try:
                # Assuming header might be like: image_name | comment_number | comment
                image_col = header.index("image_name")
                caption_col = header.index("comment")
            except ValueError:
                 print("Warning: Could not find expected headers ('image_name', 'comment') using delimiters '|'. Assuming columns 0 and 2.")
                 image_col = 0
                 caption_col = 2 # Assuming the 3rd column contains the comment

            for row in reader:
                if len(row) > max(image_col, caption_col):
                    image_name = row[image_col].strip()
                    caption = row[caption_col].strip()
                    if image_name not in captions:
                        captions[image_name] = []
                    captions[image_name].append(caption)
                else:
                    print(f"Warning: Skipping malformed row in CSV: {row}")


        # Ensure we actually got captions
        if not captions:
            print(f"Error: No captions extracted from {csv_path}. Aborting conversion.")
            return False

        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(captions, jsonfile, indent=2)

        print(f"Successfully converted captions to {json_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Captions CSV file not found at {csv_path}")
        return False
    except Exception as e:
        print(f"Error during CSV to JSON conversion: {e}")
        return False

# --- Main Preparation Function ---

def prepare_flickr30k():
    """Downloads, extracts, and prepares the Flickr30k dataset if needed."""
    if check_dataset_exists():
        print("Flickr30k dataset already found and prepared.")
        return

    print("Flickr30k dataset not found or incomplete. Starting preparation...")

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    EXTRACT_DIR.mkdir(exist_ok=True)

    # 1. Download parts
    downloaded_parts = []
    print("Downloading Flickr30k parts...")
    for i, url in enumerate(FLICKR30K_URLS):
        part_path = DOWNLOAD_DIR / f"flickr30k_part{i:02d}"
        if not download_file(url, part_path):
            print("Download failed. Cleaning up and exiting.")
            shutil.rmtree(DOWNLOAD_DIR)
            return
        downloaded_parts.append(part_path)

    # 2. Concatenate parts
    print("Concatenating downloaded parts...")
    try:
        with open(CONCATENATED_ZIP_PATH, "wb") as outfile:
            for part_path in downloaded_parts:
                with open(part_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
    except IOError as e:
        print(f"Error concatenating files: {e}")
        print("Cleaning up and exiting.")
        shutil.rmtree(DOWNLOAD_DIR)
        return

    # 3. Extract zip file
    print(f"Extracting {CONCATENATED_ZIP_PATH}...")
    try:
        with zipfile.ZipFile(CONCATENATED_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile:
        print(f"Error: Failed to unzip file. It might be corrupted.")
        print("Cleaning up and exiting.")
        shutil.rmtree(DOWNLOAD_DIR)
        return
    except Exception as e:
         print(f"Error during extraction: {e}")
         print("Cleaning up and exiting.")
         shutil.rmtree(DOWNLOAD_DIR)
         return

    # 4. Move Images
    print("Moving images...")
    source_image_dir_option1 = EXTRACT_DIR / "images" # Original assumption
    source_image_dir_option2 = EXTRACT_DIR      # Alternative: images directly in extract dir
    actual_source_dir = None

    if source_image_dir_option1.is_dir():
        actual_source_dir = source_image_dir_option1
        print(f"Found images in {actual_source_dir}")
    else:
        # Check if images are directly in the extraction folder
        image_files_found = False
        try:
            for item in source_image_dir_option2.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_files_found = True
                    break
        except FileNotFoundError:
             pass # Handle case where EXTRACT_DIR itself might not exist

        if image_files_found:
             actual_source_dir = source_image_dir_option2
             print(f"Found images directly in {actual_source_dir}")

    if actual_source_dir:
        try:
            moved_count = 0
            for item in actual_source_dir.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    target_path = IMAGE_DIR / item.name
                    shutil.move(str(item), target_path)
                    moved_count += 1
            print(f"Moved {moved_count} image files to {IMAGE_DIR}")
        except Exception as e:
            print(f"Error moving images from {actual_source_dir}: {e}")
            # Consider cleanup or exiting depending on severity
    else:
        print(f"Warning: Could not find image files in expected locations ({source_image_dir_option1} or {source_image_dir_option2}).")

    # 5. Convert Captions (CSV to JSON)
    # Find the captions CSV file (might be named results.csv)
    extracted_csv_path = EXTRACT_DIR / "results.csv" # Adjust if zip structure differs
    if extracted_csv_path.is_file():
        if not convert_csv_to_json(extracted_csv_path, CAPTIONS_JSON_PATH):
             print("Caption conversion failed. The dataset might be incomplete.")
             # Decide if we should clean up and exit
    else:
        print(f"Warning: Captions CSV file not found after extraction: {extracted_csv_path}")
        print(f"Please ensure {CAPTIONS_JSON_PATH} exists or is created manually.")

    # 6. Clean up temporary download/extraction files
    print("Cleaning up temporary files...")
    try:
        shutil.rmtree(DOWNLOAD_DIR)
    except OSError as e:
        print(f"Error removing temporary directory {DOWNLOAD_DIR}: {e}")

    print("--- Flickr30k Preparation Complete --- ")

if __name__ == "__main__":
    prepare_flickr30k() 
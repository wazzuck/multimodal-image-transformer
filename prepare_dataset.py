"""
Downloads and prepares the Flickr30k dataset.

Checks if the dataset exists in the designated 'assets' directory
(one level up from the script location). If not found, it proceeds to:
1. Download the dataset parts from a GitHub release.
2. Concatenate the downloaded parts into a single zip file.
3. Extract the contents of the zip file.
4. Move the image files to the 'assets/images' directory.
5. Convert the captions from the provided CSV format to a JSON file ('assets/captions.json').
6. Clean up temporary download and extraction files.
"""

import os
import requests
import zipfile
import shutil
import csv
import json
from pathlib import Path
from tqdm import tqdm

# Determine the script's directory to make paths relative to the script's location
# .resolve() ensures the path is absolute, .parent gets the directory containing the script.
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Configuration ---
# Define the main directory where the final dataset assets will be stored.
# It's set to be one directory level above the script's location.
DATA_DIR = SCRIPT_DIR.parent / "assets"
# Define the subdirectory within DATA_DIR where images will be stored.
IMAGE_DIR = DATA_DIR / "images"
# Define the path for the final JSON file containing captions.
CAPTIONS_JSON_PATH = DATA_DIR / "captions.json"
# Define the temporary directory for downloads and extraction.
# This directory will be created inside DATA_DIR and removed after processing.
DOWNLOAD_DIR = DATA_DIR / "temp_download"

# URLs for the different parts of the Flickr30k dataset hosted on GitHub releases.
# The dataset is split into multiple parts for easier downloading.
FLICKR30K_URLS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]
# Define the path where the downloaded parts will be concatenated into a single zip file.
CONCATENATED_ZIP_PATH = DOWNLOAD_DIR / "flickr30k.zip"
# Define the path where the contents of the concatenated zip file will be extracted.
EXTRACT_DIR = DOWNLOAD_DIR / "flickr30k_extracted"

# --- Helper Functions ---

def download_file(url, dest_path):
    """
    Downloads a file from a given URL to a destination path with a progress bar.

    Args:
        url (str): The URL of the file to download.
        dest_path (Path): The local path where the file should be saved.

    Returns:
        bool: True if download is successful, False otherwise.
    """
    print(f"Downloading {url} to {dest_path}...")
    try:
        # Make a GET request to the URL, stream=True allows downloading large files efficiently.
        # timeout is set to prevent indefinite hanging.
        response = requests.get(url, stream=True, timeout=30)
        # Raise an exception for bad status codes (like 404 Not Found or 500 Server Error).
        response.raise_for_status()
        # Get the total file size from headers, default to 0 if not available.
        total_size = int(response.headers.get('content-length', 0))
        # Define the block size for downloading chunks (1 KiB).
        block_size = 1024

        # Open the destination file in binary write mode.
        # Use tqdm to create a progress bar showing download progress.
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name, # Description shown on the progress bar (filename).
            total=total_size,   # Total size for calculating progress percentage.
            unit='iB',          # Unit for displaying size (information bytes).
            unit_scale=True,    # Automatically scale units (KiB, MiB, etc.).
            unit_divisor=1024,  # Use 1024 for KiB/MiB calculation.
        ) as bar:
            # Iterate over the response content in chunks.
            for data in response.iter_content(block_size):
                # Write the downloaded chunk to the file.
                size = f.write(data)
                # Update the progress bar with the size of the chunk written.
                bar.update(size)
        # If download completes without exceptions, return True.
        return True
    # Handle network-related errors (DNS failure, refused connection, etc.).
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    # Handle file system errors (permission denied, disk full, etc.).
    except IOError as e:
        print(f"Error writing file {dest_path}: {e}")
        return False

def check_dataset_exists():
    """
    Checks if the core components of the prepared dataset seem to exist.

    Returns:
        bool: True if the image directory and captions JSON file exist and the
              image directory is not empty, False otherwise.
    """
    print(f"Checking for existing dataset at {DATA_DIR}...")
    # Check if the main image directory exists and is a directory.
    if not IMAGE_DIR.is_dir():
        print(f"Image directory not found: {IMAGE_DIR}")
        return False
    # Check if the captions JSON file exists and is a file.
    if not CAPTIONS_JSON_PATH.is_file():
        print(f"Captions JSON file not found: {CAPTIONS_JSON_PATH}")
        return False

    # Perform a basic check to see if the image directory contains any files.
    try:
        # Attempt to get the first item from the directory iterator.
        # If the directory is empty, next() will raise StopIteration immediately.
        # any() is a more pythonic way to check if an iterator has items.
        if not any(IMAGE_DIR.iterdir()):
            print(f"Image directory exists but is empty: {IMAGE_DIR}")
            return False
        # If we reach here, the directory exists, is not empty, and the JSON exists.
        print("Found existing image directory (non-empty) and captions file.")
        return True
    # Catch the specific case where the directory is empty.
    except StopIteration:
         print(f"Image directory exists but is empty: {IMAGE_DIR}")
         return False
    # Catch potential OS errors during directory listing (e.g., permission errors).
    except OSError as e:
         print(f"Error checking directory contents {IMAGE_DIR}: {e}")
         return False

def convert_csv_to_json(csv_path, json_path):
    """
    Converts the Flickr30k captions CSV file (results.csv) to a JSON format.

    The expected CSV format is pipe-delimited ('|') with columns like:
    image_name| comment_number| comment

    The output JSON format groups captions by image filename:
    {
      "image_filename1.jpg": ["caption1", "caption2", ...],
      "image_filename2.jpg": ["captionA", "captionB", ...]
    }

    Args:
        csv_path (Path): The path to the input CSV file.
        json_path (Path): The path where the output JSON file should be saved.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    print(f"Converting captions from {csv_path} to {json_path}...")
    captions = {} # Dictionary to store captions, keyed by image name.
    try:
        # Open the CSV file for reading with UTF-8 encoding.
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Use csv.reader, trying comma as the delimiter first.
            # csv.QUOTE_MINIMAL handles quotes only when necessary (e.g., if the caption itself contains a comma).
            reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header_read = False
            image_col = -1
            caption_col = -1

            # Attempt to read the first row as a potential header
            try:
                first_row = next(reader)
                # Simple check for header content
                if len(first_row) >= 2 and 'image' in first_row[0].lower() and ('caption' in first_row[1].lower() or 'comment' in first_row[1].lower()):
                    header = [h.strip() for h in first_row]
                    image_col = 0 # Assume image is first column
                    caption_col = 1 # Assume caption is second column
                    print(f"Detected header: {header}. Using columns: image={image_col}, caption={caption_col}")
                    header_read = True
                else:
                     # If the first row doesn't look like a header, assume no header
                     print("Warning: Did not detect a standard header (e.g., 'image_name,comment'). Assuming column indices 0 (image) and 1 (caption).")
                     csvfile.seek(0) # Rewind to process the first row as data
                     reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader
                     image_col = 0
                     caption_col = 1
            except StopIteration:
                print("Warning: CSV file appears to be empty.")
                return False # Cannot proceed with an empty file
            except Exception as e:
                print(f"Warning: Error reading the first line or determining delimiter: {e}. Assuming column indices 0 (image) and 1 (caption) with comma delimiter.")
                csvfile.seek(0) # Rewind
                reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader
                image_col = 0
                caption_col = 1


            # Iterate through each row in the CSV file.
            for i, row in enumerate(reader):
                # Skip the header row if it was read
                if header_read and i == 0:
                     # This condition is actually unnecessary now as the header is read before the loop
                     # Kept conceptually, but next() already advanced the iterator
                     pass # Already processed header

                # Basic check for row integrity (has enough columns).
                if len(row) > max(image_col, caption_col):
                    # Extract image name and caption, stripping leading/trailing whitespace.
                    image_name = row[image_col].strip()
                    # Strip potential quotes and then whitespace from the caption
                    caption = row[caption_col].strip().strip('"').strip()

                    # Skip potentially problematic header-like rows if header detection failed
                    # This check might be less reliable now, remove or adjust if needed
                    # if image_col == 0 and caption_col == 1 and i == 0 and not header_read and "image" in image_name and "comment" in caption:
                    #      print("Skipping likely header row that wasn't detected initially.")
                    #      continue

                    # If the image name is not yet a key in the dictionary, add it with an empty list.
                    if image_name not in captions:
                        captions[image_name] = []
                    # Append the current caption to the list for this image.
                    captions[image_name].append(caption)
                else:
                    # Log a warning for rows that don't have the expected number of columns.
                    # Adding +1 because enumerate starts at 0, +1 if header was read
                    line_num = i + (1 if header_read else 1)
                    print(f"Warning: Skipping malformed row in CSV (line ~{line_num}): {row}. Expected at least {max(image_col, caption_col) + 1} columns.")


        # After processing the CSV, check if any captions were actually extracted.
        if not captions:
            print(f"Error: No captions extracted from {csv_path}. CSV might be empty or in an unexpected format. Aborting conversion.")
            return False

        # Open the target JSON file for writing with UTF-8 encoding.
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            # Write the captions dictionary to the JSON file.
            # indent=2 makes the output human-readable.
            json.dump(captions, jsonfile, indent=2)

        print(f"Successfully converted {len(captions)} images' captions to {json_path}")
        return True

    # Handle the case where the input CSV file doesn't exist.
    except FileNotFoundError:
        print(f"Error: Captions CSV file not found at {csv_path}")
        return False
    # Catch any other unexpected errors during the conversion process.
    except Exception as e:
        print(f"Error during CSV to JSON conversion: {e}")
        return False

# --- Main Preparation Function ---

def prepare_flickr30k():
    """
    Orchestrates the entire download, extraction, and preparation process for Flickr30k.
    Only proceeds if the dataset is not found by check_dataset_exists().
    """
    # Check if the dataset is already prepared. If so, do nothing.
    if check_dataset_exists():
        print("Flickr30k dataset already found and prepared.")
        return

    print("Flickr30k dataset not found or incomplete. Starting preparation...")

    # Ensure the necessary directories exist. Create them if they don't.
    # exist_ok=True prevents an error if the directory already exists.
    try:
        DATA_DIR.mkdir(exist_ok=True)
        IMAGE_DIR.mkdir(exist_ok=True)
        DOWNLOAD_DIR.mkdir(exist_ok=True)
        # Note: EXTRACT_DIR will be created during extraction if needed,
        # or implicitly when DOWNLOAD_DIR is created if it's the same.
        # EXTRACT_DIR.mkdir(exist_ok=True) # Usually not needed here
    except OSError as e:
        print(f"Error creating directories: {e}. Check permissions.")
        return # Cannot proceed without directories

    # --- Step 1: Download dataset parts ---
    downloaded_parts = [] # Keep track of successfully downloaded part file paths.
    print("--- Step 1: Downloading Flickr30k parts ---")
    all_downloads_successful = True
    for i, url in enumerate(FLICKR30K_URLS):
        # Construct the local path for the current part.
        part_path = DOWNLOAD_DIR / f"flickr30k_part{i:02d}"
        # Attempt to download the file.
        if not download_file(url, part_path):
            print(f"Download failed for {url}.")
            all_downloads_successful = False
            break # Stop downloading if one part fails.
        downloaded_parts.append(part_path)

    # If any download failed, clean up and exit early.
    if not all_downloads_successful:
        print("One or more downloads failed. Cleaning up temporary files and exiting.")
        try:
            shutil.rmtree(DOWNLOAD_DIR)
        except OSError as e:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e}")
        return

    # --- Step 2: Concatenate downloaded parts ---
    print(f"--- Step 2: Concatenating {len(downloaded_parts)} downloaded parts into {CONCATENATED_ZIP_PATH} ---")
    try:
        # Open the target concatenated zip file in binary write mode.
        with open(CONCATENATED_ZIP_PATH, "wb") as outfile:
            # Iterate through the paths of the downloaded parts.
            for part_path in downloaded_parts:
                print(f"Appending {part_path.name}...")
                # Open each part file in binary read mode.
                with open(part_path, "rb") as infile:
                    # Efficiently copy the content from the part file to the output file.
                    shutil.copyfileobj(infile, outfile)
        print(f"Successfully concatenated parts into {CONCATENATED_ZIP_PATH}")
    # Handle potential file I/O errors during concatenation.
    except IOError as e:
        print(f"Error concatenating files: {e}")
        print("Cleaning up temporary files and exiting.")
        try:
            shutil.rmtree(DOWNLOAD_DIR)
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return

    # --- Step 3: Extract the zip file ---
    print(f"--- Step 3: Extracting {CONCATENATED_ZIP_PATH} to {EXTRACT_DIR} ---")
    try:
        # Ensure the target extraction directory exists.
        EXTRACT_DIR.mkdir(exist_ok=True)
        # Open the concatenated zip file for reading.
        with zipfile.ZipFile(CONCATENATED_ZIP_PATH, 'r') as zip_ref:
            # Extract all members of the zip archive to the EXTRACT_DIR.
            print(f"Extracting members: {zip_ref.namelist()}")
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Successfully extracted files to {EXTRACT_DIR}")
    # Handle errors indicating the zip file is corrupted or not a valid zip file.
    except zipfile.BadZipFile:
        print(f"Error: Failed to unzip file {CONCATENATED_ZIP_PATH}. It might be corrupted or incomplete.")
        print("Cleaning up temporary files and exiting.")
        try:
            shutil.rmtree(DOWNLOAD_DIR)
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return
    # Catch other potential exceptions during extraction (e.g., disk full, permissions).
    except Exception as e:
         print(f"Error during extraction: {e}")
         print("Cleaning up temporary files and exiting.")
         try:
            shutil.rmtree(DOWNLOAD_DIR)
         except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
         return

    # --- Step 4: Move Images to final location ---
    print(f"--- Step 4: Moving images to {IMAGE_DIR} ---")
    # Define potential source directories for images within the extracted files.
    source_image_dir_option1 = EXTRACT_DIR / "Images"      # Check for 'Images' (uppercase I) based on ls output
    source_image_dir_option2 = EXTRACT_DIR                 # e.g., flickr30k_extracted/
    actual_source_dir = None # Will store the directory where images are actually found.

    # Check if the 'Images' subdirectory exists.
    if source_image_dir_option1.is_dir():
        actual_source_dir = source_image_dir_option1
        print(f"Found images in standard subdirectory: {actual_source_dir}")
    else:
        # If 'Images' subdirectory not found, check for image files directly in EXTRACT_DIR.
        print(f"Did not find standard subdirectory {source_image_dir_option1}. Checking base extraction directory {source_image_dir_option2}...")
        image_files_found_in_base = False
        try:
            # Iterate through items in the base extraction directory.
            for item in source_image_dir_option2.iterdir():
                # Check if the item is a file and has a common image extension.
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_files_found_in_base = True
                    print(f"Found image file {item.name} directly in {source_image_dir_option2}.")
                    break # Found at least one, no need to check further.
        except FileNotFoundError:
             # This might happen if EXTRACT_DIR itself couldn't be created or accessed.
             print(f"Warning: Extraction directory {source_image_dir_option2} not found or accessible.")
             pass

        # If image files were found directly in the base extraction directory, set it as the source.
        if image_files_found_in_base:
             actual_source_dir = source_image_dir_option2
             print(f"Confirmed images are located directly in: {actual_source_dir}")

    # If a source directory containing images was identified:
    if actual_source_dir:
        try:
            moved_count = 0 # Counter for moved images.
            print(f"Iterating through {actual_source_dir} to move image files...")
            # Iterate through all items in the identified source directory.
            for item in actual_source_dir.iterdir():
                # Check again if it's a file with an image extension (handles nested non-image files).
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Define the full path for the destination file.
                    target_path = IMAGE_DIR / item.name
                    # Move the file from the source to the target directory.
                    # Using str() conversion as shutil.move might prefer strings.
                    shutil.move(str(item), target_path)
                    moved_count += 1
            print(f"Successfully moved {moved_count} image files to {IMAGE_DIR}")
        # Handle potential errors during the file moving process (permissions, disk space).
        except Exception as e:
            print(f"Error moving images from {actual_source_dir} to {IMAGE_DIR}: {e}")
            # Consider whether to halt the process or just warn.
            # Depending on the error, dataset might be incomplete.
    else:
        # If no images were found in either expected location.
        print(f"Warning: Could not find image files in expected locations ({source_image_dir_option1} or {source_image_dir_option2}). Cannot move images.")
        # This indicates a potential problem with the extracted archive structure.

    # --- Step 5: Convert Captions (CSV to JSON) ---
    print(f"--- Step 5: Converting captions CSV to JSON ({CAPTIONS_JSON_PATH}) ---")
    # Define the expected path for the captions file within the extracted data.
    # Based on ls output, the file is captions.txt
    extracted_captions_path = EXTRACT_DIR / "captions.txt"
    print(f"Looking for captions file at: {extracted_captions_path}")

    # Check if the expected captions file exists.
    if extracted_captions_path.is_file():
        # Attempt to convert the found captions file to the target JSON format.
        # Assuming captions.txt has the same format as results.csv (pipe-delimited)
        if not convert_csv_to_json(extracted_captions_path, CAPTIONS_JSON_PATH):
             # If conversion fails, log a warning. The dataset might be usable without captions,
             # or this might indicate a critical failure depending on the use case.
             print("Warning: Caption conversion failed. The final dataset might be incomplete or lack captions.")
             # Decide if we should clean up and exit based on severity
    else:
        # If the captions file wasn't found where expected.
        print(f"Warning: Captions file not found after extraction at expected path: {extracted_captions_path}")
        print(f"Cannot create {CAPTIONS_JSON_PATH}. Please ensure the captions file exists manually or check the archive contents.")

    # --- Step 6: Clean up temporary files ---
    print(f"--- Step 6: Cleaning up temporary download/extraction files in {DOWNLOAD_DIR} ---")
    try:
        # Recursively remove the temporary download directory and all its contents.
        # This includes the downloaded parts, the concatenated zip, and the extracted files.
        shutil.rmtree(DOWNLOAD_DIR)
        print(f"Successfully removed temporary directory: {DOWNLOAD_DIR}")
    # Handle errors that might occur during cleanup (e.g., files in use, permissions).
    except OSError as e:
        print(f"Warning: Error removing temporary directory {DOWNLOAD_DIR}: {e}")
    print("--- Flickr30k Preparation Process Complete --- ")
    # Final check to inform the user if essential parts seem missing
    if not IMAGE_DIR.is_dir() or not any(IMAGE_DIR.iterdir()):
         print(f"Warning: Post-processing check indicates the image directory {IMAGE_DIR} is missing or empty.")
    if not CAPTIONS_JSON_PATH.is_file():
         print(f"Warning: Post-processing check indicates the captions file {CAPTIONS_JSON_PATH} is missing.")


# Standard Python entry point check.
# Ensures that the prepare_flickr30k() function is called only when the script
# is executed directly (e.g., `python prepare_dataset.py`), not when it's imported as a module.
if __name__ == "__main__":
    print("Running Flickr30k dataset preparation script...")
    prepare_flickr30k()
    print("Script finished.") 
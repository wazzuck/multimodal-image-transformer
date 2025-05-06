"""
Downloads and prepares the Flickr30k dataset based on paths defined in config.py.

Checks if the dataset exists in the designated directories from config.py.
If not found, it proceeds to:
1. Download the dataset parts from a GitHub release.
2. Concatenate the downloaded parts into a single zip file.
3. Extract the contents of the zip file.
4. Move the image files to the configured IMAGE_DIR.
5. Convert the captions from the provided CSV format to the configured CAPTIONS_FILE (JSON).
6. Clean up temporary download and extraction files.
"""

import os
import requests
import zipfile
import shutil
import csv
import json
from pathlib import Path # Import Path
from tqdm import tqdm
import config # Import the config module

# --- Configuration from config.py ---
# Use paths defined in the config module, converting them to Path objects
# Ensure config.py defines: DATA_DIR, IMAGE_DIR, CAPTIONS_FILE
# We derive temporary paths relative to the configured DATA_DIR

try:
    # Convert config string paths to Path objects
    CONFIG_DATA_DIR = Path(config.DATA_DIR).resolve() # Use resolve to make it absolute
    IMAGE_DIR = Path(config.IMAGE_DIR).resolve()
    CAPTIONS_JSON_PATH = Path(config.CAPTIONS_FILE).resolve()

    # Define temporary directories relative to the resolved DATA_DIR from config
    # This keeps temporary files within the specified data area
    DOWNLOAD_DIR = CONFIG_DATA_DIR / "temp_download"
    CONCATENATED_ZIP_PATH = DOWNLOAD_DIR / "flickr30k.zip"
    EXTRACT_DIR = DOWNLOAD_DIR / "flickr30k_extracted"

    # Create the base data directory if it doesn't exist
    CONFIG_DATA_DIR.mkdir(parents=True, exist_ok=True)

except AttributeError as e:
    print(f"Error: Missing necessary path definition in config.py: {e}")
    print("Please ensure config.py defines DATA_DIR, IMAGE_DIR, and CAPTIONS_FILE.")
    # Optionally exit or raise here if configuration is critical
    exit(1) # Exit if config is incomplete
except Exception as e:
    print(f"Error processing paths from config.py: {e}")
    exit(1)


# URLs for the different parts of the Flickr30k dataset hosted on GitHub releases.
# The dataset is split into multiple parts for easier downloading.
FLICKR30K_URLS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]


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
        response = requests.get(url, stream=True, timeout=60) # Increased timeout
        # Raise an exception for bad status codes (like 404 Not Found or 500 Server Error).
        response.raise_for_status()
        # Get the total file size from headers, default to 0 if not available.
        total_size = int(response.headers.get('content-length', 0))
        # Define the block size for downloading chunks (1 KiB).
        block_size = 1024

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

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
    Checks if the core components of the prepared dataset seem to exist
    using paths from config.py.

    Returns:
        bool: True if the image directory and captions JSON file exist and the
              image directory is not empty, False otherwise.
    """
    print(f"Checking for existing dataset using config paths:")
    print(f"  Image Directory: {IMAGE_DIR}")
    print(f"  Captions JSON: {CAPTIONS_JSON_PATH}")

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
    Converts the Flickr30k captions CSV file (results.csv or captions.txt) to a JSON format.

    The output JSON format groups captions by image filename:
    {
      "image_filename1.jpg": ["caption1", "caption2", ...],
      "image_filename2.jpg": ["captionA", "captionB", ...]
    }

    Args:
        csv_path (Path): The path to the input CSV file (e.g., captions.txt from archive).
        json_path (Path): The path where the output JSON file should be saved (from config).

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    print(f"Converting captions from {csv_path} to {json_path}...")
    captions = {} # Dictionary to store captions, keyed by image name.
    try:
        # Open the CSV file for reading with UTF-8 encoding.
        # Determine delimiter based on common Flickr30k formats
        delimiter_to_use = ',' # Default to comma
        try:
             with open(csv_path, 'r', encoding='utf-8') as f_test:
                  first_line = f_test.readline()
                  if '|' in first_line and first_line.count('|') >= 2:
                       delimiter_to_use = '|'
                       print("Detected pipe ('|') delimiter in captions file.")
                  else:
                       print("Using comma (',') delimiter for captions file.")
        except Exception as e_delim:
             print(f"Warning: Could not automatically determine delimiter for {csv_path}: {e_delim}. Defaulting to comma.")


        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Use the determined delimiter
            reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header_read = False
            image_col = -1
            caption_col = -1

            # Attempt to read the first row as a potential header
            try:
                first_row = next(reader)
                # Adapt header check for different delimiters
                cols = [h.strip().lower() for h in first_row]
                expected_image_keys = ['image', 'image_name']
                expected_caption_keys = ['comment', 'caption']

                found_image = False
                found_caption = False

                for idx, col_name in enumerate(cols):
                     if any(key in col_name for key in expected_image_keys):
                          image_col = idx
                          found_image = True
                     if any(key in col_name for key in expected_caption_keys):
                          caption_col = idx
                          found_caption = True

                if found_image and found_caption:
                    print(f"Detected header: {cols}. Using columns: image={image_col}, caption={caption_col}")
                    header_read = True
                else:
                     # If the first row doesn't look like a header, assume no header
                     print(f"Warning: Did not detect a standard header in '{first_row}'. Assuming column indices 0 (image) and {1 if delimiter_to_use == '|' else 2} (caption) based on delimiter.")
                     csvfile.seek(0) # Rewind to process the first row as data
                     reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader
                     image_col = 0
                     # Common formats:
                     # image|number|caption (pipe) -> caption is col 2
                     # image,caption (comma) -> caption is col 1
                     caption_col = 2 if delimiter_to_use == '|' else 1 # Adjust assumed caption column index

            except StopIteration:
                print("Warning: CSV file appears to be empty.")
                return False # Cannot proceed with an empty file
            except Exception as e:
                print(f"Warning: Error reading the first line or determining header/delimiter: {e}. Assuming column indices 0 (image) and {1 if delimiter_to_use == '|' else 2} (caption).")
                csvfile.seek(0) # Rewind
                reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader
                image_col = 0
                caption_col = 2 if delimiter_to_use == '|' else 1

            # Iterate through each row in the CSV file.
            for i, row in enumerate(reader):
                # Basic check for row integrity (has enough columns).
                try:
                    if len(row) > max(image_col, caption_col):
                        # Extract image name and caption, stripping leading/trailing whitespace.
                        image_name = row[image_col].strip()
                        # Strip potential quotes and then whitespace from the caption
                        caption = row[caption_col].strip().strip('"').strip()

                        # If the image name is not yet a key in the dictionary, add it with an empty list.
                        if image_name not in captions:
                            captions[image_name] = []
                        # Append the current caption to the list for this image.
                        captions[image_name].append(caption)
                    else:
                        # Log a warning for rows that don't have the expected number of columns.
                        line_num = i + (1 if header_read else 1)
                        print(f"Warning: Skipping malformed row in CSV (line ~{line_num}): {row}. Expected at least {max(image_col, caption_col) + 1} columns based on header/assumption.")
                except IndexError:
                     line_num = i + (1 if header_read else 1)
                     print(f"Warning: Skipping row with IndexError in CSV (line ~{line_num}): {row}. Column index out of bounds (image={image_col}, caption={caption_col}).")


        # After processing the CSV, check if any captions were actually extracted.
        if not captions:
            print(f"Error: No captions extracted from {csv_path}. CSV might be empty or in an unexpected format. Aborting conversion.")
            return False

        # Ensure the directory for the JSON file exists
        json_path.parent.mkdir(parents=True, exist_ok=True)

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
        import traceback
        traceback.print_exc() # Print detailed traceback
        return False


# --- Main Preparation Function ---

def prepare_flickr30k():
    """
    Orchestrates the entire download, extraction, and preparation process for Flickr30k,
    using paths defined in config.py.
    Only proceeds if the dataset is not found by check_dataset_exists().
    """
    # Check if the dataset is already prepared using config paths.
    if check_dataset_exists():
        print("Flickr30k dataset already found and prepared according to config paths.")
        return

    print("Flickr30k dataset not found or incomplete based on config paths. Starting preparation...")

    # Ensure the necessary directories exist (using paths derived from config).
    # Create them if they don't. parents=True creates intermediate dirs.
    try:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # EXTRACT_DIR is within DOWNLOAD_DIR, will be created if needed or by DOWNLOAD_DIR creation.
    except OSError as e:
        print(f"Error creating directories ({IMAGE_DIR}, {DOWNLOAD_DIR}): {e}. Check permissions and paths in config.py.")
        return # Cannot proceed without directories

    # --- Step 1: Download dataset parts ---
    downloaded_parts = [] # Keep track of successfully downloaded part file paths.
    print("--- Step 1: Downloading Flickr30k parts ---")
    all_downloads_successful = True
    for i, url in enumerate(FLICKR30K_URLS):
        # Construct the local path for the current part inside DOWNLOAD_DIR.
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
            if DOWNLOAD_DIR.exists():
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
            if DOWNLOAD_DIR.exists():
                 shutil.rmtree(DOWNLOAD_DIR)
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return

    # --- Step 3: Extract the zip file ---
    print(f"--- Step 3: Extracting {CONCATENATED_ZIP_PATH} to {EXTRACT_DIR} ---")
    try:
        # Ensure the target extraction directory exists.
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        # Open the concatenated zip file for reading.
        with zipfile.ZipFile(CONCATENATED_ZIP_PATH, 'r') as zip_ref:
            # Extract all members of the zip archive to the EXTRACT_DIR.
            print(f"Extracting members...") # Removed namelist() for brevity on potentially long lists
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Successfully extracted files to {EXTRACT_DIR}")
    # Handle errors indicating the zip file is corrupted or not a valid zip file.
    except zipfile.BadZipFile:
        print(f"Error: Failed to unzip file {CONCATENATED_ZIP_PATH}. It might be corrupted or incomplete.")
        print("Cleaning up temporary files and exiting.")
        try:
            if DOWNLOAD_DIR.exists():
                 shutil.rmtree(DOWNLOAD_DIR)
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return
    # Catch other potential exceptions during extraction (e.g., disk full, permissions).
    except Exception as e:
         print(f"Error during extraction: {e}")
         print("Cleaning up temporary files and exiting.")
         try:
             if DOWNLOAD_DIR.exists():
                  shutil.rmtree(DOWNLOAD_DIR)
         except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
         return

    # --- Step 4: Move Images to final location (IMAGE_DIR from config) ---
    print(f"--- Step 4: Moving images to configured directory: {IMAGE_DIR} ---")
    # Define potential source directories for images within the extracted files.
    # Common structures: flickr30k_extracted/Images/ or flickr30k_extracted/flickr30k-images/ or just files directly
    source_image_dir_options = [
        EXTRACT_DIR / "Images",      # Check for 'Images' (uppercase I)
        EXTRACT_DIR / "flickr30k-images", # Check for 'flickr30k-images'
        EXTRACT_DIR # Check base directory as last resort
    ]
    actual_source_dir = None # Will store the directory where images are actually found.

    # Find the actual directory containing image files
    for potential_dir in source_image_dir_options:
        if potential_dir.is_dir():
             print(f"Checking potential image source directory: {potential_dir}...")
             try:
                 # Check if it actually contains image files before declaring it the source
                 contains_images = False
                 for item in potential_dir.iterdir():
                      if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                           contains_images = True
                           print(f"Found images in: {potential_dir}")
                           break # Found images, this is our source
                 if contains_images:
                     actual_source_dir = potential_dir
                     break # Found the source, stop checking other options
             except OSError as e:
                 print(f"Warning: Could not read contents of {potential_dir}: {e}")
        # If it's the base EXTRACT_DIR, we only check it if others weren't found/valid
        elif potential_dir == EXTRACT_DIR:
             print(f"Checking base extraction directory {EXTRACT_DIR} for images...")
             try:
                 contains_images = False
                 for item in potential_dir.iterdir():
                      # Only consider files directly in EXTRACT_DIR, not subdirs checked earlier
                      if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                           contains_images = True
                           print(f"Found images directly in: {EXTRACT_DIR}")
                           break
                 if contains_images:
                      actual_source_dir = EXTRACT_DIR
                      # Don't break here, let loop finish in case a dedicated subdir exists later
             except OSError as e:
                  print(f"Warning: Could not read contents of {EXTRACT_DIR}: {e}")


    # If a source directory containing images was identified:
    if actual_source_dir:
        print(f"Moving images from {actual_source_dir} to {IMAGE_DIR}...")
        try:
            moved_count = 0 # Counter for moved images.
            skipped_count = 0 # Counter for skipped (existing) images
            # Iterate through all items in the identified source directory.
            for item in actual_source_dir.iterdir():
                # Check again if it's a file with an image extension (handles nested non-image files).
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Define the full path for the destination file using IMAGE_DIR from config.
                    target_path = IMAGE_DIR / item.name
                    # Move the file from the source to the target directory.
                    # Only move if it doesn't exist in the target to avoid errors if run partially before
                    if not target_path.exists():
                        # Using str() conversion as shutil.move might prefer strings.
                        shutil.move(str(item), str(target_path))
                        moved_count += 1
                    else:
                        # print(f"Skipping existing file: {target_path}") # Optional: noisy
                        skipped_count += 1
            print(f"Successfully moved {moved_count} image files to {IMAGE_DIR}. Skipped {skipped_count} existing files.")
        # Handle potential errors during the file moving process (permissions, disk space).
        except Exception as e:
            print(f"Error moving images from {actual_source_dir} to {IMAGE_DIR}: {e}")
            # Consider whether to halt the process or just warn.
    else:
        # If no images were found in any expected location.
        print(f"Warning: Could not find source directory containing image files within {EXTRACT_DIR}. Expected structures like 'Images/', 'flickr30k-images/', or files directly inside. Cannot move images.")
        # This indicates a potential problem with the extracted archive structure.

    # --- Step 5: Convert Captions (CSV to JSON, using CAPTIONS_JSON_PATH from config) ---
    print(f"--- Step 5: Converting captions CSV to JSON ({CAPTIONS_JSON_PATH}) ---")
    # Define the expected path for the captions file within the extracted data.
    # Common names: results.csv, captions.txt
    extracted_captions_path_options = [
        EXTRACT_DIR / "results.csv",
        EXTRACT_DIR / "captions.txt"
    ]
    actual_captions_path = None

    for potential_path in extracted_captions_path_options:
        if potential_path.is_file():
            actual_captions_path = potential_path
            print(f"Found captions file at: {actual_captions_path}")
            break

    # If a captions file was found:
    if actual_captions_path:
        # Attempt to convert the found captions file to the target JSON format (CAPTIONS_JSON_PATH from config).
        if not convert_csv_to_json(actual_captions_path, CAPTIONS_JSON_PATH):
             # If conversion fails, log a warning.
             print(f"Warning: Caption conversion failed for {actual_captions_path}. The final captions file {CAPTIONS_JSON_PATH} might be missing or incomplete.")
             # Decide if we should clean up and exit based on severity
    else:
        # If the captions file wasn't found where expected.
        print(f"Warning: Captions file not found after extraction. Looked for {extracted_captions_path_options}")
        print(f"Cannot create {CAPTIONS_JSON_PATH}. Please ensure a captions file (e.g., results.csv, captions.txt) exists in the archive or provide it manually.")

    # --- Step 6: Clean up temporary files ---
    print(f"--- Step 6: Cleaning up temporary download/extraction files in {DOWNLOAD_DIR} ---")
    try:
        # Recursively remove the temporary download directory and all its contents.
        if DOWNLOAD_DIR.exists():
            shutil.rmtree(DOWNLOAD_DIR)
            print(f"Successfully removed temporary directory: {DOWNLOAD_DIR}")
        else:
            print(f"Temporary directory {DOWNLOAD_DIR} not found, skipping removal.")
    # Handle errors that might occur during cleanup (e.g., files in use, permissions).
    except OSError as e:
        print(f"Warning: Error removing temporary directory {DOWNLOAD_DIR}: {e}")

    print("--- Flickr30k Preparation Process Complete --- ")
    # Final check to inform the user if essential parts seem missing based on config paths
    if not IMAGE_DIR.is_dir() or not any(IMAGE_DIR.iterdir()):
         print(f"Warning: Post-processing check indicates the configured image directory {IMAGE_DIR} is missing or empty.")
    if not CAPTIONS_JSON_PATH.is_file():
         print(f"Warning: Post-processing check indicates the configured captions file {CAPTIONS_JSON_PATH} is missing.")


# Standard Python entry point check.
# Ensures that the prepare_flickr30k() function is called only when the script
# is executed directly (e.g., `python prepare_dataset.py`), not when it's imported as a module.
if __name__ == "__main__":
    print("Running Flickr30k dataset preparation script using paths from config.py...")
    prepare_flickr30k()
    print("Script finished.") 
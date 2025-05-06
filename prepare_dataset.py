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
    # Convert string paths from config.py to Path objects for robust path manipulation.
    # .resolve() makes the paths absolute, ensuring consistency regardless of execution location.
    CONFIG_DATA_DIR = Path(config.DATA_DIR).resolve() # Base data directory from config.
    IMAGE_DIR = Path(config.IMAGE_DIR).resolve()      # Final directory for images from config.
    CAPTIONS_JSON_PATH = Path(config.CAPTIONS_FILE).resolve() # Final path for JSON captions from config.

    # Define temporary directories relative to the resolved DATA_DIR from config.
    # This centralizes temporary files within the main data area specified by the user.
    DOWNLOAD_DIR = CONFIG_DATA_DIR / "temp_download"          # For storing downloaded archive parts.
    CONCATENATED_ZIP_PATH = DOWNLOAD_DIR / "flickr30k.zip"  # Path for the combined zip file.
    EXTRACT_DIR = DOWNLOAD_DIR / "flickr30k_extracted"    # For extracting the contents of the zip file.

    # Create the base data directory if it doesn't already exist.
    # parents=True: Creates any necessary parent directories.
    # exist_ok=True: Does not raise an error if the directory already exists.
    CONFIG_DATA_DIR.mkdir(parents=True, exist_ok=True)

except AttributeError as e:
    # This error occurs if config.py is missing one of the required path variables.
    print(f"Error: Missing necessary path definition in config.py: {e}")
    print("Please ensure config.py defines DATA_DIR, IMAGE_DIR, and CAPTIONS_FILE.")
    exit(1) # Exit the script as configuration is critical for its operation.
except Exception as e:
    # Catches any other errors during path processing (e.g., invalid path formats).
    print(f"Error processing paths from config.py: {e}")
    exit(1) # Exit as paths are fundamental.


# URLs for the different parts of the Flickr30k dataset hosted on GitHub releases.
# The dataset is split into multiple parts for easier downloading and management.
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
        # Make a GET request to the URL.
        # stream=True: Allows downloading the file in chunks, efficient for large files.
        # timeout=60: Sets a 60-second timeout for the request to prevent indefinite hanging.
        response = requests.get(url, stream=True, timeout=60) # Increased timeout
        # Raise an HTTPError for bad responses (4xx or 5xx status codes).
        response.raise_for_status()
        # Get the total file size from the 'content-length' header. Defaults to 0 if not present.
        total_size = int(response.headers.get('content-length', 0))
        # Define the block size for downloading chunks (1024 bytes = 1 KiB).
        block_size = 1024

        # Ensure the destination directory for the downloaded file exists.
        # parents=True creates parent directories if they don't exist.
        # exist_ok=True doesn't raise an error if the directory already exists.
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the destination file in binary write mode ("wb").
        # Use tqdm to create a progress bar for monitoring the download.
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name, # Description displayed on the progress bar (usually the filename).
            total=total_size,   # Total size of the download, for calculating progress percentage.
            unit='iB',          # Unit for displaying size (iB = information Bytes).
            unit_scale=True,    # Automatically scale units (e.g., KiB, MiB).
            unit_divisor=1024,  # Base for unit scaling (1024 for binary units like KiB/MiB).
        ) as bar:
            # Iterate over the response content in chunks of 'block_size'.
            for data in response.iter_content(block_size):
                # Write the downloaded chunk to the file.
                size = f.write(data)
                # Update the progress bar by the amount of data written in this chunk.
                bar.update(size)
        # If the download loop completes without exceptions, the download was successful.
        return True
    # Handle network-related errors (e.g., DNS failure, connection refused).
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    # Handle file system errors during writing (e.g., permission denied, disk full).
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

    # Check if the configured IMAGE_DIR exists and is actually a directory.
    if not IMAGE_DIR.is_dir():
        print(f"Image directory not found: {IMAGE_DIR}")
        return False
    # Check if the configured CAPTIONS_JSON_PATH exists and is actually a file.
    if not CAPTIONS_JSON_PATH.is_file():
        print(f"Captions JSON file not found: {CAPTIONS_JSON_PATH}")
        return False

    # Perform a basic check to ensure the image directory is not empty.
    try:
        # IMAGE_DIR.iterdir() creates an iterator over the contents of the directory.
        # any() checks if this iterator yields at least one item (i.e., the directory is not empty).
        # This is more efficient than listing all files if we only need to know if it's non-empty.
        if not any(IMAGE_DIR.iterdir()):
            print(f"Image directory exists but is empty: {IMAGE_DIR}")
            return False
        # If all checks pass, the dataset is considered to exist and be minimally valid.
        print("Found existing image directory (non-empty) and captions file.")
        return True
    # This exception occurs if iterdir() is called on an empty directory and next() is used,
    # or if other iteration issues arise, though less common with any().
    # More robust for 'any()' is simply checking its result. This catch is more for completeness
    # if a different iteration method was used.
    except StopIteration: # Should not be strictly necessary with `any()`
         print(f"Image directory exists but is empty: {IMAGE_DIR}")
         return False
    # Catch potential OS-level errors during directory listing (e.g., permission denied).
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
    captions = {} # Initialize an empty dictionary to store captions, keyed by image filename.
    try:
        # Open the CSV file for reading, specifying UTF-8 encoding which is common.
        # Determine delimiter based on common Flickr30k formats (results.csv usually uses '|', others ',')
        delimiter_to_use = ',' # Default to comma delimiter.
        try:
             # Attempt to read the first line to infer the delimiter.
             with open(csv_path, 'r', encoding='utf-8') as f_test:
                  first_line = f_test.readline()
                  # If pipe characters are present and seem to delimit columns, use pipe.
                  if '|' in first_line and first_line.count('|') >= 2: # Check for at least two pipe delimiters.
                       delimiter_to_use = '|'
                       print("Detected pipe ('|') delimiter in captions file.")
                  else:
                       print("Using comma (',') delimiter for captions file.")
        except Exception as e_delim:
             # If delimiter detection fails, print a warning and proceed with the default (comma).
             print(f"Warning: Could not automatically determine delimiter for {csv_path}: {e_delim}. Defaulting to comma.")


        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Create a CSV reader object using the determined delimiter.
            # quotechar='"' handles fields that might be enclosed in double quotes.
            # quoting=csv.QUOTE_MINIMAL specifies that quotes are only used when necessary.
            reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header_read = False # Flag to track if a header row was successfully processed.
            image_col = -1      # Initialize column index for image names.
            caption_col = -1    # Initialize column index for captions.

            # Attempt to read the first row to detect if it's a header.
            try:
                first_row = next(reader) # Read the first row from the CSV.
                # Clean and normalize column names from the first row.
                cols = [h.strip().lower() for h in first_row]
                # Define common keywords for image name and caption columns.
                expected_image_keys = ['image', 'image_name']
                expected_caption_keys = ['comment', 'caption']

                found_image = False   # Flag for finding the image column.
                found_caption = False # Flag for finding the caption column.

                # Iterate through the column names from the first row.
                for idx, col_name in enumerate(cols):
                     # Check if any of the expected image keys are part of the current column name.
                     if any(key in col_name for key in expected_image_keys):
                          image_col = idx
                          found_image = True
                     # Check if any of the expected caption keys are part of the current column name.
                     if any(key in col_name for key in expected_caption_keys):
                          caption_col = idx
                          found_caption = True

                if found_image and found_caption:
                    # If both image and caption columns are identified, assume it was a header.
                    print(f"Detected header: {cols}. Using columns: image={image_col}, caption={caption_col}")
                    header_read = True
                else:
                     # If a clear header isn't found, issue a warning and fall back to default column indices.
                     print(f"Warning: Did not detect a standard header in '{first_row}'. Assuming column indices 0 (image) and {1 if delimiter_to_use == '|' else 2} (caption) based on delimiter.")
                     csvfile.seek(0) # Rewind the file to re-read the first row as data.
                     reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader.
                     image_col = 0 # Assume image name is in the first column.
                     # Adjust assumed caption column index based on the detected/defaulted delimiter.
                     # Flickr30k format with '|' often has image|index|caption (caption is 3rd col, index 2).
                     # Flickr30k format with ',' might be image,caption (caption is 2nd col, index 1).
                     caption_col = 2 if delimiter_to_use == '|' else 1

            except StopIteration:
                # This occurs if the CSV file is empty.
                print("Warning: CSV file appears to be empty.")
                return False # Cannot proceed with an empty file.
            except Exception as e:
                # Catch any other errors during header/delimiter processing.
                print(f"Warning: Error reading the first line or determining header/delimiter: {e}. Assuming column indices 0 (image) and {1 if delimiter_to_use == '|' else 2} (caption).")
                csvfile.seek(0) # Rewind the file.
                reader = csv.reader(csvfile, delimiter=delimiter_to_use, quotechar='"', quoting=csv.QUOTE_MINIMAL) # Re-initialize reader.
                image_col = 0   # Fallback image column index.
                caption_col = 2 if delimiter_to_use == '|' else 1 # Fallback caption column index.

            # Iterate through each row in the CSV file (starting after the header if one was read).
            for i, row in enumerate(reader):
                # Basic check for row integrity: ensure it has enough columns for the determined indices.
                try:
                    if len(row) > max(image_col, caption_col):
                        # Extract image name and caption from their respective columns.
                        # .strip() removes leading/trailing whitespace.
                        image_name = row[image_col].strip()
                        # .strip('"') removes potential surrounding quotes from the caption, then strip whitespace.
                        caption = row[caption_col].strip().strip('"').strip()

                        # If the image name is not yet a key in the 'captions' dictionary, add it with an empty list.
                        if image_name not in captions:
                            captions[image_name] = []
                        # Append the current caption to the list of captions for this image.
                        captions[image_name].append(caption)
                    else:
                        # Log a warning for rows that don't have the expected number of columns.
                        line_num = i + (1 if header_read else 1) # Adjust line number if header was skipped.
                        print(f"Warning: Skipping malformed row in CSV (line ~{line_num}): {row}. Expected at least {max(image_col, caption_col) + 1} columns based on header/assumption.")
                except IndexError:
                     # This occurs if a row is too short for the expected image_col or caption_col.
                     line_num = i + (1 if header_read else 1)
                     print(f"Warning: Skipping row with IndexError in CSV (line ~{line_num}): {row}. Column index out of bounds (image={image_col}, caption={caption_col}).")


        # After processing the entire CSV, check if any captions were actually extracted.
        if not captions:
            print(f"Error: No captions extracted from {csv_path}. CSV might be empty or in an unexpected format. Aborting conversion.")
            return False

        # Ensure the directory for the output JSON file exists.
        # parents=True creates parent directories if needed. exist_ok=True avoids error if it exists.
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the target JSON file in write mode with UTF-8 encoding.
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            # Write the 'captions' dictionary to the JSON file.
            # indent=2 formats the JSON output with an indentation of 2 spaces for readability.
            json.dump(captions, jsonfile, indent=2)

        print(f"Successfully converted {len(captions)} images' captions to {json_path}")
        return True

    # Handle the case where the input CSV file does not exist.
    except FileNotFoundError:
        print(f"Error: Captions CSV file not found at {csv_path}")
        return False
    # Catch any other unexpected errors during the conversion process.
    except Exception as e:
        print(f"Error during CSV to JSON conversion: {e}")
        import traceback # Import traceback for detailed error reporting.
        traceback.print_exc() # Print the full traceback for debugging.
        return False


# --- Main Preparation Function ---

def prepare_flickr30k():
    """
    Orchestrates the entire download, extraction, and preparation process for Flickr30k,
    using paths defined in config.py.
    Only proceeds if the dataset is not found by check_dataset_exists().
    """
    # Check if the dataset is already prepared according to paths specified in config.py.
    if check_dataset_exists():
        print("Flickr30k dataset already found and prepared according to config paths.")
        return # If found, no further action is needed.

    print("Flickr30k dataset not found or incomplete based on config paths. Starting preparation...")

    # Ensure the necessary target directories (IMAGE_DIR from config) and temporary
    # directories (DOWNLOAD_DIR) exist. Create them if they don't.
    # parents=True: Creates any necessary parent directories.
    # exist_ok=True: Does not raise an error if the directory already exists.
    try:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)      # Final image storage.
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)   # Temporary download location.
        # EXTRACT_DIR is created later, before extraction, as it's within DOWNLOAD_DIR.
    except OSError as e:
        print(f"Error creating directories ({IMAGE_DIR}, {DOWNLOAD_DIR}): {e}. Check permissions and paths in config.py.")
        return # Cannot proceed without these directories.

    # --- Step 1: Download dataset parts ---
    downloaded_parts = [] # List to keep track of successfully downloaded part file paths.
    print("--- Step 1: Downloading Flickr30k parts ---")
    all_downloads_successful = True # Flag to track if all parts download successfully.
    for i, url in enumerate(FLICKR30K_URLS):
        # Construct the local path for the current part within the DOWNLOAD_DIR.
        # f"{i:02d}" formats the number 'i' with leading zeros (e.g., 00, 01, 02).
        part_path = DOWNLOAD_DIR / f"flickr30k_part{i:02d}"
        # Attempt to download the file using the helper function.
        if not download_file(url, part_path):
            print(f"Download failed for {url}.")
            all_downloads_successful = False # Set flag to False if any download fails.
            break # Stop downloading further parts if one fails.
        downloaded_parts.append(part_path) # Add path of successfully downloaded part to list.

    # If any download failed, clean up temporary files and exit the preparation process.
    if not all_downloads_successful:
        print("One or more downloads failed. Cleaning up temporary files and exiting.")
        try:
            if DOWNLOAD_DIR.exists(): # Check if DOWNLOAD_DIR was created.
                 shutil.rmtree(DOWNLOAD_DIR) # Recursively remove DOWNLOAD_DIR and its contents.
        except OSError as e:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e}")
        return

    # --- Step 2: Concatenate downloaded parts ---
    # This step combines the multiple downloaded parts into a single .zip file.
    print(f"--- Step 2: Concatenating {len(downloaded_parts)} downloaded parts into {CONCATENATED_ZIP_PATH} ---")
    try:
        # Open the target concatenated zip file in binary write mode ("wb").
        with open(CONCATENATED_ZIP_PATH, "wb") as outfile:
            # Iterate through the paths of the downloaded parts.
            for part_path in downloaded_parts:
                print(f"Appending {part_path.name}...")
                # Open each part file in binary read mode ("rb").
                with open(part_path, "rb") as infile:
                    # Efficiently copy the content from the current part file (infile)
                    # to the end of the concatenated output file (outfile).
                    shutil.copyfileobj(infile, outfile)
        print(f"Successfully concatenated parts into {CONCATENATED_ZIP_PATH}")
    # Handle potential file I/O errors during concatenation (e.g., disk full, permissions).
    except IOError as e:
        print(f"Error concatenating files: {e}")
        print("Cleaning up temporary files and exiting.")
        try:
            if DOWNLOAD_DIR.exists():
                 shutil.rmtree(DOWNLOAD_DIR) # Clean up on error.
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return

    # --- Step 3: Extract the zip file ---
    # This step extracts the contents of the concatenated .zip file.
    print(f"--- Step 3: Extracting {CONCATENATED_ZIP_PATH} to {EXTRACT_DIR} ---")
    try:
        # Ensure the target extraction directory exists.
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        # Open the concatenated zip file in read mode ('r').
        with zipfile.ZipFile(CONCATENATED_ZIP_PATH, 'r') as zip_ref:
            # Extract all members (files and directories) from the zip archive
            # into the specified EXTRACT_DIR.
            print(f"Extracting members...")
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Successfully extracted files to {EXTRACT_DIR}")
    # Handle errors indicating the zip file is corrupted or not a valid zip format.
    except zipfile.BadZipFile:
        print(f"Error: Failed to unzip file {CONCATENATED_ZIP_PATH}. It might be corrupted or incomplete.")
        print("Cleaning up temporary files and exiting.")
        try:
            if DOWNLOAD_DIR.exists():
                 shutil.rmtree(DOWNLOAD_DIR) # Clean up on error.
        except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
        return
    # Catch other potential exceptions during extraction (e.g., disk full, file permission issues).
    except Exception as e:
         print(f"Error during extraction: {e}")
         print("Cleaning up temporary files and exiting.")
         try:
             if DOWNLOAD_DIR.exists():
                  shutil.rmtree(DOWNLOAD_DIR) # Clean up on error.
         except OSError as e_clean:
            print(f"Error during cleanup of {DOWNLOAD_DIR}: {e_clean}")
         return

    # --- Step 4: Move Images to final location (IMAGE_DIR from config) ---
    # This step locates image files within the extracted content and moves them to the
    # final destination directory specified in config.py.
    print(f"--- Step 4: Moving images to configured directory: {IMAGE_DIR} ---")
    # Define a list of potential subdirectory names where images might be located
    # within the extracted archive. This handles variations in archive structure.
    source_image_dir_options = [
        EXTRACT_DIR / "Images",           # Common: 'Images' subdirectory.
        EXTRACT_DIR / "flickr30k-images", # Another common name.
        EXTRACT_DIR                       # Fallback: check the root of EXTRACT_DIR.
    ]
    actual_source_dir = None # Variable to store the path of the directory where images are found.

    # Iterate through the potential source directories to find where the images are located.
    for potential_dir in source_image_dir_options:
        if potential_dir.is_dir(): # Check if the potential directory exists.
             print(f"Checking potential image source directory: {potential_dir}...")
             try:
                 # Check if this directory actually contains image files before declaring it the source.
                 contains_images = False
                 for item in potential_dir.iterdir(): # Iterate over items in the directory.
                      # Check if the item is a file and has a common image file extension.
                      if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                           contains_images = True
                           print(f"Found images in: {potential_dir}")
                           break # Found images, no need to check further items in this directory.
                 if contains_images:
                     actual_source_dir = potential_dir # Set this as the actual source directory.
                     break # Found the image source directory, exit the loop.
             except OSError as e:
                 # Warn if there's an error reading the contents (e.g., permissions).
                 print(f"Warning: Could not read contents of {potential_dir}: {e}")
        # Special handling for checking the base EXTRACT_DIR itself for images.
        # This is usually a fallback if no dedicated image subdirectory is found.
        elif potential_dir == EXTRACT_DIR: # This case is mostly a fallback.
             print(f"Checking base extraction directory {EXTRACT_DIR} for images...")
             try:
                 contains_images = False
                 for item in potential_dir.iterdir():
                      # Ensure we only consider files directly in EXTRACT_DIR, not subdirectories
                      # that might have been (or will be) checked by other options.
                      if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                           contains_images = True
                           print(f"Found images directly in: {EXTRACT_DIR}")
                           break
                 if contains_images:
                      actual_source_dir = EXTRACT_DIR
                      # Note: Loop continues here to allow a more specific subdirectory (like "Images")
                      # to be chosen if it exists and also contains images.
             except OSError as e:
                  print(f"Warning: Could not read contents of {EXTRACT_DIR}: {e}")


    # If a source directory containing images was identified:
    if actual_source_dir:
        print(f"Moving images from {actual_source_dir} to {IMAGE_DIR}...")
        try:
            moved_count = 0   # Counter for successfully moved images.
            skipped_count = 0 # Counter for images skipped because they already exist at the destination.
            # Iterate through all items (files and directories) in the identified source image directory.
            for item in actual_source_dir.iterdir():
                # Double-check if the item is a file and has an image extension.
                # This handles cases where non-image files might be present in the source dir.
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Define the full path for the destination file in the target IMAGE_DIR.
                    target_path = IMAGE_DIR / item.name
                    # Move the file from the source to the target directory.
                    # Only move if the file doesn't already exist at the target to avoid errors
                    # if the script was run partially before or if there are duplicate names.
                    if not target_path.exists():
                        # shutil.move requires string paths for broader compatibility, though Path objects often work.
                        shutil.move(str(item), str(target_path))
                        moved_count += 1
                    else:
                        # Optionally log skipped files, though can be noisy for many files.
                        # print(f"Skipping existing file: {target_path}")
                        skipped_count += 1
            print(f"Successfully moved {moved_count} image files to {IMAGE_DIR}. Skipped {skipped_count} existing files.")
        # Handle potential errors during the file moving process (e.g., permissions, disk space issues).
        except Exception as e:
            print(f"Error moving images from {actual_source_dir} to {IMAGE_DIR}: {e}")
            # Depending on severity, one might choose to halt or just warn. Here it's a warning.
    else:
        # If no directory containing images was found after checking all options.
        print(f"Warning: Could not find source directory containing image files within {EXTRACT_DIR}. Expected structures like 'Images/', 'flickr30k-images/', or files directly inside. Cannot move images.")
        # This indicates a potential problem with the archive's structure or content.

    # --- Step 5: Convert Captions (CSV to JSON, using CAPTIONS_JSON_PATH from config) ---
    # This step finds the captions file (usually CSV or TXT) in the extracted data and
    # converts it to the JSON format specified by CAPTIONS_JSON_PATH from config.py.
    print(f"--- Step 5: Converting captions CSV to JSON ({CAPTIONS_JSON_PATH}) ---")
    # Define a list of common names for the captions file within the extracted archive.
    extracted_captions_path_options = [
        EXTRACT_DIR / "results.csv",  # A common name for Flickr30k captions.
        EXTRACT_DIR / "captions.txt"  # Another possible name.
    ]
    actual_captions_path = None # Variable to store the path of the found captions file.

    # Iterate through the potential paths to find the actual captions file.
    for potential_path in extracted_captions_path_options:
        if potential_path.is_file(): # Check if the file exists at this path.
            actual_captions_path = potential_path
            print(f"Found captions file at: {actual_captions_path}")
            break # Captions file found, no need to check other options.

    # If a captions file was found:
    if actual_captions_path:
        # Attempt to convert the found captions file to the target JSON format
        # using the convert_csv_to_json helper function.
        if not convert_csv_to_json(actual_captions_path, CAPTIONS_JSON_PATH):
             # If conversion fails, log a warning. The script will continue but captions might be missing.
             print(f"Warning: Caption conversion failed for {actual_captions_path}. The final captions file {CAPTIONS_JSON_PATH} might be missing or incomplete.")
             # Depending on how critical captions are, one might choose to exit here.
    else:
        # If the captions file was not found in any of the expected locations.
        print(f"Warning: Captions file not found after extraction. Looked for {extracted_captions_path_options}")
        print(f"Cannot create {CAPTIONS_JSON_PATH}. Please ensure a captions file (e.g., results.csv, captions.txt) exists in the archive or provide it manually.")

    # --- Step 6: Clean up temporary files ---
    # This step removes the temporary download and extraction directory (DOWNLOAD_DIR)
    # and all its contents to free up space.
    print(f"--- Step 6: Cleaning up temporary download/extraction files in {DOWNLOAD_DIR} ---")
    try:
        # Recursively remove the DOWNLOAD_DIR. This deletes the directory and everything inside it.
        if DOWNLOAD_DIR.exists(): # Check if the directory exists before attempting to remove.
            shutil.rmtree(DOWNLOAD_DIR)
            print(f"Successfully removed temporary directory: {DOWNLOAD_DIR}")
        else:
            print(f"Temporary directory {DOWNLOAD_DIR} not found, skipping removal.")
    # Handle errors that might occur during cleanup (e.g., files in use, permission issues).
    except OSError as e:
        print(f"Warning: Error removing temporary directory {DOWNLOAD_DIR}: {e}")

    print("--- Flickr30k Preparation Process Complete --- ")
    # Perform a final check to inform the user if essential dataset components
    # (as defined in config.py) seem to be missing or empty after the process.
    if not IMAGE_DIR.is_dir() or not any(IMAGE_DIR.iterdir()):
         print(f"Warning: Post-processing check indicates the configured image directory {IMAGE_DIR} is missing or empty.")
    if not CAPTIONS_JSON_PATH.is_file():
         print(f"Warning: Post-processing check indicates the configured captions file {CAPTIONS_JSON_PATH} is missing.")


# Standard Python entry point check.
# This ensures that the prepare_flickr30k() function is called only when
# the script is executed directly (e.g., via `python prepare_dataset.py`),
# and not when it's imported as a module into another script.
if __name__ == "__main__":
    print("Running Flickr30k dataset preparation script using paths from config.py...")
    prepare_flickr30k() # Execute the main preparation function.
    print("Script finished.") 
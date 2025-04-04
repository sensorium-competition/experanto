import numpy as np
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import yaml
import typing

# Ensure correct working directory if running from a different location
# script_dir = Path(__file__).parent.absolute()
# project_root = script_dir.parents[1] # Adjusted from [2] to [1]
# os.chdir(project_root)
os.chdir("/src/code/Enigma/Libraries/PRS/tensorfabrik") # User override
print(f"Changed working directory to: {os.getcwd()}")

# --- Metadata Combining Function (from experanto/interpolators.py) ---
def combine_screen_metadatas(screen_folder: typing.Union[str, Path]) -> None:
    """
    Combines individual screen trial metadata YAML files found in the 'meta'
    subdirectory of the screen_folder into a single 'combined_meta.json' file
    in the screen_folder.

    Args:
        screen_folder: The path to the session's screen data folder (e.g., .../session_xyz/screen).
    """
    screen_folder = Path(screen_folder)
    meta_dir = screen_folder / "meta"
    output_path = screen_folder / "combined_meta.json" # Output to screen folder

    if not meta_dir.is_dir():
        # print(f"Meta directory not found in {screen_folder}, skipping.")
        raise FileNotFoundError(f"Meta directory not found: {meta_dir}")

    # Function to check if a file is a numbered yml file
    def is_numbered_yml(file_name):
        return re.fullmatch(r"\d{5}\.yml", file_name) is not None

    # Initialize an empty dictionary to store all contents
    all_data = {}

    # Get meta files and sort by number
    try:
        meta_files = [
            f
            for f in meta_dir.iterdir()
            if f.is_file() and is_numbered_yml(f.name)
        ]
        if not meta_files:
            # print(f"No numbered YAML files found in {meta_dir}, skipping combination.")
            # Create an empty JSON file if no YAMLs are found
            with open(output_path, 'w') as file:
                 json.dump({}, file)
            return # No files to combine

        meta_files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

        # Read each YAML file and store under its filename stem
        for meta_file in meta_files:
            with open(meta_file, 'r') as file:
                file_base_name = meta_file.stem
                try:
                    yaml_content = yaml.safe_load(file)
                    # Handle potential None content if YAML is empty
                    if yaml_content is None:
                        yaml_content = {}
                    all_data[file_base_name] = yaml_content
                except yaml.YAMLError as e:
                    print(f"Warning: Error reading YAML file {meta_file}: {e}, skipping file.")
                    # Decide if you want to skip this file or raise an error
                    continue # Skip to the next file

        # Write the combined data to the output JSON file
        with open(output_path, 'w') as file:
            json.dump(all_data, file, indent=4) # Added indent for readability
        # print(f"Successfully combined metadata for {screen_folder} into {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {screen_folder}: {e}")
        # Re-raise the exception to be caught by the calling script
        raise
# --- End of Metadata Combining Function ---


# Define the base directory containing session folders
# --- IMPORTANT: SET YOUR BASE DIRECTORY HERE ---
BASE_DATA_DIR = "/mnt/stor02/enigma/full_foundation_export"
# ---

if not os.path.isdir(BASE_DATA_DIR):
    print(f"Error: Base data directory not found: {BASE_DATA_DIR}")
    exit(1)

# List all subdirectories in the base directory
folders = [os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, f))]

if not folders:
    print(f"No session folders found in {BASE_DATA_DIR}")
    exit(1)

print(f"Found {len(folders)} potential session folders.")

def process_folder(folder_path):
    """Calls the combine_screen_metadatas function for a single folder."""
    try:
        # We only need the screen metadata for combination
        screen_folder_path = Path(folder_path) / "screen"
        if screen_folder_path.is_dir():
            combine_screen_metadatas(screen_folder_path) # Call the local function
            return folder_path, True, None # Indicate success
        else:
            # print(f"Screen subfolder not found in {folder_path}, skipping.")
            return folder_path, False, "Screen subfolder not found" # Indicate skipped
    except FileNotFoundError as e: # Specifically catch if meta dir is missing
         # print(f"Skipping {folder_path}: {e}")
         return folder_path, False, str(e) # Indicate skipped due to missing meta
    except Exception as e:
        # print(f"Error processing {folder_path}: {e}")
        return folder_path, False, str(e) # Indicate failure with error message

# Use ThreadPoolExecutor to run process_folder in parallel
# --- ADJUST number of threads as needed ---
num_threads = 10 # Increased threads, adjust based on IO/CPU limits
# ---
results = []
failed_folders_details = {}
skipped_folders_details = {} # Separate dict for skipped folders

print(f"Processing folders using {num_threads} threads...")
# Use tqdm with the executor to keep the progress bar
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all tasks
    futures = {executor.submit(process_folder, folder): folder for folder in folders}
    # Process results as they complete, updating tqdm
    for future in tqdm(as_completed(futures), total=len(folders), ncols=100, ascii=True, desc="Combining Metadata"):
        folder, success, message = future.result()
        results.append((folder, success))
        if not success:
             if message == "Screen subfolder not found" or "Meta directory not found" in message:
                 skipped_folders_details[folder] = message
             else:
                 failed_folders_details[folder] = message


# Summarize results
successful_folders = [res[0] for res in results if res[1]]
failed_folders = list(failed_folders_details.keys())
skipped_folders = list(skipped_folders_details.keys())


print(f"\nProcessed {len(folders)} folders.")
print(f"Successfully combined metadata for: {len(successful_folders)} folders.")
print(f"Skipped (no screen/meta folder): {len(skipped_folders)} folders.")
# Optionally print details for skipped folders
# if skipped_folders:
#     print("Skipped folders details:")
#     for folder, reason in skipped_folders_details.items():
#         print(f"- {folder}: {reason}")
if failed_folders:
    print(f"Failed folders ({len(failed_folders)}):")
    for folder, error in failed_folders_details.items():
        print(f"- {folder}: {error}")

print("\nMetadata combination process finished.")

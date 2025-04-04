import numpy as np
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
print(os.getcwd())
os.chdir("/src/code/Enigma/Libraries/PRS/tensorfabrik")


folders = [os.path.join("/mnt/stor02/enigma/full_foundation_export", f) for f in os.listdir("/mnt/stor02/enigma/full_foundation_export") if os.path.isdir(os.path.join("/mnt/stor02/enigma/full_foundation_export", f))]

from experanto.configs import DEFAULT_CONFIG as cfg
from experanto.datasets import ChunkDataset

def process_folder(folder):
    # This function contains the work done for each folder
    try:
        dataset = ChunkDataset(folder, global_sampling_rate=cfg.dataset.global_sampling_rate, global_chunk_size=cfg.dataset.global_chunk_size, modality_config=cfg.dataset.modality_config)
        # You might want to do something with the dataset here or return a status
        return folder, True # Indicate success
    except Exception as e:
        print(f"Error processing {folder}: {e}")
        return folder, False # Indicate failure

# Use ThreadPoolExecutor to run process_folder in parallel
num_threads = 5
results = []

# Use tqdm with the executor to keep the progress bar
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all tasks
    futures = [executor.submit(process_folder, folder) for folder in folders]
    # Process results as they complete, updating tqdm
    for future in tqdm(as_completed(futures), total=len(folders), ncols=100, ascii=True):
        results.append(future.result())

# Optional: Process results if needed
successful_folders = [res[0] for res in results if res[1]]
failed_folders = [res[0] for res in results if not res[1]]

print(f"\nProcessed {len(folders)} folders.")
print(f"Successfully processed: {len(successful_folders)}")
if failed_folders:
    print(f"Failed folders ({len(failed_folders)}):")
    for folder in failed_folders:
        print(f"- {folder}")
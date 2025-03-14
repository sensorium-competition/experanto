#!/usr/bin/env python
"""
Distributed dataloader performance testing script for PyTorch DDP.
This script profiles dataloader performance across multiple GPUs.


echo "Starting distributed dataloader testing with $NUM_PROCESSES processes"
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_PROCESSES \
  --master_port=$MASTER_PORT \
  run_distributed_dataloaders.py

"""
import rootutils
# Handles initialization of project root dir, see `rootutils<https://github.com/ashleve/rootutils>`__.
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import logging
import time
import datetime
import os
import os.path as path
import numpy as np
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from experanto.dataloaders import get_multisession_dataloader, LongCycler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device before anything else
    torch.cuda.set_device(local_rank)

    # Initialize process group with timeout
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    logger.info(f"Initialized process group: rank {rank}/{world_size}, local_rank: {local_rank}")

    # NCCL test
    tensor = torch.tensor([rank], dtype=torch.int64, device=f"cuda:{local_rank}")
    gathered = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}")
                for _ in range(world_size)]

    # Add explicit barrier before all_gather
    logger.info(f"Rank {rank}: Before barrier")
    dist.barrier()
    logger.info(f"Rank {rank}: After barrier, before all_gather")

    dist.all_gather(gathered, tensor)
    gathered = [t.item() for t in gathered]

    logger.info(f"Rank {rank}: Can see processes {gathered}")
    return rank, local_rank, world_size


def get_dataset_paths(rank, datasets_per_rank, all_paths):
    """
    Distribute dataset paths across ranks.
    Each rank gets datasets_per_rank consecutive paths.
    """
    start_idx = rank * datasets_per_rank
    end_idx = min(start_idx + datasets_per_rank, len(all_paths))

    # If we run out of datasets, wrap around
    if start_idx >= len(all_paths):
        start_idx = start_idx % len(all_paths)
        end_idx = min(start_idx + datasets_per_rank, len(all_paths))

    paths = all_paths[start_idx:end_idx]

    # wrap around when there are more ranks than datasets
    if len(paths) < datasets_per_rank:
        additional_needed = datasets_per_rank - len(paths)
        paths = np.concatenate([paths, all_paths[:additional_needed]])

    return paths


def gather_dataset_info(dataloader, rank, world_size):
    """
    Gather dataset keys and neuron counts from all distributed dataloaders.

    Args:
        dataloader: The local dataloader
        rank: Process rank
        world_size: Total number of processes

    Returns:
        dict: Combined dictionary with all keys and their neuron counts
    """
    # Get local dataset info
    local_info = {}
    for k, v in dataloader.loaders.items():
        try:
            batch = next(iter(v))
            n_neurons = batch["responses"].shape[-1]
            local_info[k] = n_neurons
        except StopIteration:
            # Handle empty dataloader case
            logger.warning(f"Rank {rank}: Empty dataloader for key {k}")
            continue

    # Convert local info to a serializable format
    # [(key1, neurons1), (key2, neurons2), ...]
    local_items = list(local_info.items())

    # Determine maximum number of items across all ranks
    item_counts = [torch.tensor([len(local_items)], device="cuda") for _ in range(world_size)]
    dist.all_gather(item_counts, torch.tensor([len(local_items)], device="cuda"))
    max_items = max([count.item() for count in item_counts])

    # Pad local items list if needed
    if len(local_items) < max_items:
        # Use a special padding value that we can filter out later
        local_items.extend([("__pad__", -1) for _ in range(max_items - len(local_items))])

    # For each item, gather its key and neuron count
    all_keys = []
    all_neurons = []

    for i in range(max_items):
        if i < len(local_items):
            key, neurons = local_items[i]
            key_tensor = torch.tensor([ord(c) for c in key] + [0] * (50 - len(key)), device="cuda")
            neuron_tensor = torch.tensor([neurons], device="cuda")
        else:
            key_tensor = torch.tensor([0] * 50, device="cuda")
            neuron_tensor = torch.tensor([-1], device="cuda")

        # Gather keys and neurons from all ranks
        gathered_keys = [torch.zeros(50, dtype=torch.long, device="cuda") for _ in range(world_size)]
        gathered_neurons = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]

        dist.all_gather(gathered_keys, key_tensor)
        dist.all_gather(gathered_neurons, neuron_tensor)

        # Convert back to strings and add to our lists
        for r in range(world_size):
            key_chars = [chr(c.item()) for c in gathered_keys[r] if c.item() > 0]
            key = ''.join(key_chars)
            neurons = gathered_neurons[r].item()

            if key != "__pad__" and neurons != -1:
                all_keys.append(key)
                all_neurons.append(neurons)

    # Combine into a dictionary
    combined_info = {k: n for k, n in zip(all_keys, all_neurons)}

    if rank == 0:
        logger.info(f"Gathered dataset info: {combined_info}")

    return combined_info


def sync_dataloader_lengths(dataloader, rank, world_size):
    """
    Synchronize dataloader lengths across all ranks.

    Args:
        dataloader: The dataloader to synchronize
        rank: Process rank
        world_size: Total number of processes

    Returns:
        int: The synchronized number of batches per epoch
    """
    # Get local dataloader length in batches
    local_length = len(dataloader)

    # Create tensors to hold lengths from all ranks
    all_lengths = [torch.tensor([0], device="cuda") for _ in range(world_size)]

    # Gather all lengths
    dist.all_gather(all_lengths, torch.tensor([local_length], device="cuda"))
    all_lengths = [length.item() for length in all_lengths]

    # Find minimum length across all ranks
    min_length = min(all_lengths)

    if rank == 0:
        logger.info(f"Dataset lengths across ranks (in batches): {all_lengths}")
        logger.info(f"Using synchronized length of {min_length} batches per epoch")

    return min_length


class SyncedLongCycler(LongCycler):
    """
    Extends LongCycler to support distributed training by syncing the number of batches
    across all ranks while preserving the cycling behavior for loaders of unequal size.
    """

    def __init__(self, loaders, max_batches):
        super().__init__(loaders)
        # Override max_batches with the synchronized value
        self.max_batches = max_batches
        # Store total iterations per epoch (all keys * batches)
        self.iterations_per_epoch = len(self.loaders) * self.max_batches

    def __len__(self):
        return self.iterations_per_epoch


def prepare_synced_dataloaders(dataloaders, rank, world_size):
    """
    Prepares dataloaders for distributed training by synchronizing lengths.

    Args:
        dataloaders: Dictionary of dataloaders for this rank
        rank: Process rank
        world_size: Total number of processes

    Returns:
        SyncedLongCycler: Synchronized cycler for training
    """
    # Find the minimum length across all dataloaders on this rank
    local_max_length = max([len(loader) for loader in dataloaders.values()])

    # Synchronize this minimum length across all ranks
    all_max_lengths = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(all_max_lengths, torch.tensor([local_max_length], device="cuda"))
    all_max_lengths = [length.item() for length in all_max_lengths]

    # Use the global minimum as the synchronized length
    global_max_length = max(all_max_lengths)

    if rank == 0:
        logger.info(f"Minimum dataloader lengths across ranks: {all_max_lengths}")
        logger.info(f"Using synchronized length of {global_max_length} batches per key per epoch")

    # Create a synced cycler with the synchronized batch count
    return SyncedLongCycler(dataloaders, global_max_length)


def profile_dataloader(dataloader, cfg, max_batches=5000, dtype=torch.bfloat16, log_every=10):
    """
    Profile the performance of a dataloader with synchronized length across ranks.

    Args:
        dataloader: The dataloader to profile
        cfg: Configuration object
        max_batches: Maximum number of batches to process
        dtype: Data type for tensors
        log_every: Interval (in seconds) to log dataloader throughput

    Returns:
        float: Overall throughput in batches/second
    """
    # Get distributed information
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    frames_per_batch = cfg.dataloader.batch_size * cfg.dataset.modality_config.screen.chunk_size
    logger.info(f"Rank {rank}: Profiling dataloader with {frames_per_batch} frames per batch")

    # Performance tracking variables
    start_time = time.time()
    last_report_time = start_time
    batches_since_last_report = 0

    for i, (key, batch) in tqdm(
            enumerate(dataloader),
            total=max_batches,
            disable=rank != 0,
            mininterval=1.0,
    ):
        if i >= max_batches:
            break

        # Transfer to GPU
        videos = batch["screen"].to("cuda", dtype, non_blocking=True).transpose(1, 2)
        responses = batch["responses"].to("cuda", dtype, non_blocking=True)

        # Track performance
        batches_since_last_report += 1
        current_time = time.time()
        time_since_last_report = current_time - last_report_time

        # Report throughput every 10 seconds
        if time_since_last_report >= log_every:
            throughput = (batches_since_last_report / time_since_last_report) * frames_per_batch
            logger.info(
                f"Rank {rank}: Throughput: {throughput/1000:.1f}k frames/second (over last {time_since_last_report:.2f}s)")

            # Reset counters
            batches_since_last_report = 0
            last_report_time = current_time

    # Print final statistics
    total_time = time.time() - start_time
    overall_throughput = ((i + 1) * frames_per_batch) / total_time
    logger.info(f"Rank {rank}: Overall throughput: {frames_per_batch/1000:.1f}k frames/second over {total_time:.2f}s")

    # Gather statistics from all ranks
    throughputs = [torch.tensor([0.0], device="cuda") for _ in range(world_size)]
    dist.all_gather(throughputs, torch.tensor([overall_throughput], device="cuda"))

    if rank == 0:
        throughputs = [t.item() for t in throughputs]
        avg_throughput = sum(throughputs) / len(throughputs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)

        logger.info(f"===== Distributed Performance Summary =====")
        logger.info(f"Average throughput across all ranks: {avg_throughput:.2f} batches/second")
        logger.info(f"Min throughput: {min_throughput:.2f}, Max throughput: {max_throughput:.2f}")
        logger.info(f"Throughput imbalance: {(max_throughput - min_throughput) / avg_throughput * 100:.2f}%")

    return overall_throughput


@hydra.main(config_path=f"{root}/configs", config_name="benchmarking", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for the distributed dataloader profiling."""

    # Set up the distributed environment
    rank, local_rank, world_size = setup_distributed()

    # Print configuration on rank 0
    if rank == 0:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Get paths from config
    full_paths = np.array([os.path.join(cfg.datapath.root, f) for f in cfg.datapath.files])

    # Get paths for this rank
    rank_paths = get_dataset_paths(
        rank,
        cfg.distributed.datasets_per_rank,
        full_paths
    )
    print(f"Rank {rank}: Dataset paths: {rank_paths}")

    # Create dataloader
    logger.info(f"creating dataloader on rank {rank}")
    train_dl = get_multisession_dataloader(
        rank_paths,
        cfg,
    )
    logger.info(f"finished dataloader on rank {rank}")

    if rank == 0:
        logger.info("Waiting for all processes to sync the dataloaders ...")
    dist.barrier()
    # Gather dataset info from all ranks
    if rank == 0:
        logger.info(f"== Gathering dataset_info ==")
    dataset_info = gather_dataset_info(train_dl, rank, world_size)

    if rank == 0:
        logger.info(f"== Gathering full dataset info complete ==")
        logger.info(f"Dataset info: {dataset_info}")

    # Create a synced cycler for training
    synced_dl = prepare_synced_dataloaders(train_dl.loaders, rank, world_size)

    # Profile dataloader
    throughput = profile_dataloader(
        dataloader=synced_dl,
        cfg=cfg,
        max_batches=cfg.distributed.max_batches,
    )

    # Clean up
    logger.info(f"Rank {rank}: Waiting for all processes to complete before cleanup...")
    dist.barrier()

    # Sleep a bit to ensure all processes have completed their barrier
    time.sleep(1)

    if rank == 0:
        logger.info("All processes completed. Starting cleanup...")

    # Clean up in a controlled manner
    try:
        # Ensure all CUDA operations are complete
        torch.cuda.synchronize()

        # Clean up the process group
        dist.destroy_process_group()
        logger.info(f"Rank {rank}: Successfully cleaned up process group")
    except Exception as e:
        logger.error(f"Rank {rank}: Error during cleanup: {str(e)}")

    # Final message
    logger.info(f"Rank {rank}: Finished profiling and cleanup complete")


if __name__ == "__main__":
    main()
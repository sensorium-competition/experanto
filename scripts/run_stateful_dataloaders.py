import torch
import numpy as np
from torch.utils.data import Dataset
from experanto.utils import StatefulDataLoader, StatefulLongCycler

class DummyDataset(Dataset):
    """Dataset that returns labeled elements (e.g., A_0, A_1, etc.)"""
    def __init__(self, size=100, prefix='A'):
        self.size = size
        self.prefix = prefix
        # Create data like ['A_0', 'A_1', ...] for prefix 'A'
        self.data = [f"{prefix}_{i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_stateful_dataloader():
    # Create dataset and dataloader
    dataset = DummyDataset(size=100, prefix='X')
    dataloader = StatefulDataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True,
        seed=42
    )
    
    # Collect first half of the epoch normally
    first_half = []
    state = None
    for i, batch in enumerate(dataloader):

        if i <= 5:
            first_half.append(batch)

            if i == 5:  # Save state halfway through
                state = dataloader.get_state()            
        else:
            break
    
    # Create new dataloader and set state
    new_dataloader = StatefulDataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True,
        seed=42
    )
    new_dataloader.set_state(state)
    
    # Collect remaining batches
    second_half = []
    for batch in new_dataloader:
        second_half.append(batch)
    
    # Now collect full epoch without interruption
    full_dataloader = StatefulDataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True,
        seed=42
    )
    full_epoch = [batch for batch in full_dataloader]
    
    # Compare results
    # assert len(first_half + second_half) == len(full_epoch)

    print("first_half + second_half = ")
    print(first_half)
    print(second_half)
    print("full_epoch = ")
    print(full_epoch[:len(first_half)])
    print(full_epoch[len(first_half):])

    
    for (a, b) in zip(first_half + second_half, full_epoch):
        assert all(x == y for x, y in zip(a, b)), f"Interrupted and full runs differ!\n{a}\n{b}"
    print("StatefulDataLoader test passed!")

def test_stateful_long_cycler():
    # Create multiple datasets with different prefixes
    datasets = {
        'A': DummyDataset(20, prefix='A'),
        'B': DummyDataset(10, prefix='B'),
        'C': DummyDataset(5, prefix='C')
    }
    
    dataloaders = {
        name: StatefulDataLoader(dataset, batch_size=5, shuffle=True, seed=42)
        for name, dataset in datasets.items()
    }
    
    # Create cycler
    cycler = StatefulLongCycler(dataloaders, seed=47)
    initial_state = cycler.get_state()

    # Collect first portion of iterations
    first_portion = []
    state = None
    print("\nFirst portion (interrupted run):")
    for i, (key, batch) in enumerate(cycler):

        if i <= 10:
            first_portion.append((key, batch))
            print(f"Iteration {i}: Dataset {key}, Batch: {batch}")

            if i == 10:  # Save state after 10 iterations
                state = cycler.get_state()            
        else:
            break
    
    # Create new cycler and set state
    new_cycler = StatefulLongCycler(dataloaders, seed=42)
    new_cycler.set_state(state)
    
    # Collect remaining portion
    second_portion = []
    print("\nSecond portion (after state restoration):")
    for i, (key, batch) in enumerate(new_cycler):
        second_portion.append((key, batch))
        print(f"Iteration {i}: Dataset {key}, Batch: {batch}")
    
    # Now collect full run without interruption
    full_cycler = StatefulLongCycler(dataloaders, seed=42)
    full_cycler.set_state(initial_state)
    print("\nFull run (uninterrupted):")
    full_run = []
    for i, (key, batch) in enumerate(full_cycler):
        full_run.append((key, batch))
        print(f"Iteration {i}: Dataset {key}, Batch: {batch}")
    
    # Compare results
    assert len(first_portion + second_portion) == len(full_run)
    for i, ((key1, batch1), (key2, batch2)) in enumerate(zip(first_portion + second_portion, full_run)):
        assert key1 == key2, f"Keys differ at position {i}! {key1} != {key2}"
        assert batch1 == batch2, f"Batches differ at position {i}!\n{batch1}\n{batch2}"
    print("\nStatefulLongCycler test passed!")

if __name__ == "__main__":
    test_stateful_dataloader()
    test_stateful_long_cycler()
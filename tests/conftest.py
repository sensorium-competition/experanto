"""
Pytest configuration and fixtures for testing experanto.utils functionality.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing data loaders."""
    class MockDataset(Dataset):
        def __init__(self, min_length=5, max_length=105, seed=None, session_name=None):
            self.np_rng = np.random.RandomState(seed)
            self.torch_rng = torch.Generator()
            if seed is not None:
                self.torch_rng.manual_seed(seed)
            self.length = self.np_rng.randint(min_length, max_length + 1)
            
            # Use provided session name or generate one in format #####-#-#
            if session_name is not None:
                self.session_name = session_name
            else:
                animal_id = self.np_rng.randint(10000, 99999)
                month = self.np_rng.randint(1, 12)
                day = self.np_rng.randint(1, 28)
                self.session_name = f"{animal_id}-{month}-{day}"
            self.data_key = self.session_name
            
            # Pre-generate all batches
            self.batches = [self._gen_batch() for _ in range(self.length)]
        
        def _gen_batch(self):
            # Generate data tensors
            responses = torch.randn(60, 4096, generator=self.torch_rng) * 25 + 25
            screen = torch.randn(1, 60, 36, 64, generator=self.torch_rng) * 2
            eye_tracker = torch.randn(40, 4, generator=self.torch_rng)
            treadmill = torch.rand(40, 1, generator=self.torch_rng) * 0.4 + 0.25
            # Generate timestamps (relative, starting from 0)
            timestamps = {
                'responses': torch.rand(60, 4096, generator=self.torch_rng) * 2.0,
                'screen': torch.rand(60, generator=self.torch_rng) * 2.0,
                'eye_tracker': torch.rand(40, generator=self.torch_rng) * 2.0,
                'treadmill': torch.rand(40, generator=self.torch_rng) * 2.0
            }
            # Sort timestamps to be monotonic
            for key in timestamps:
                timestamps[key], _ = torch.sort(timestamps[key], dim=-1)
            # Construct batch data
            batch_data = {
                    'responses': responses,
                    'screen': screen,
                    'eye_tracker': eye_tracker,
                    'treadmill': treadmill,
                    'timestamps': timestamps
                }
            return batch_data
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            return self.batches[idx]
        
        def get_state(self):
            pass

        def set_state(self, state):
            pass

        def reset_state(self):
            pass
    
    return MockDataset
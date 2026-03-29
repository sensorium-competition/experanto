# Experanto

Experanto is a Python package designed for interpolating recordings and stimuli in neuroscience experiments. It enables users to load single or multiple experiments and create efficient dataloaders for machine learning applications.

> [!IMPORTANT]
> If you're interested in contributing or looking for a **Google Summer of Code** project, we discuss ideas and planned features in this [ideas thread](https://github.com/sensorium-competition/experanto/discussions/101) — please share your thoughts there before opening a PR.

## Features

- **Unified Experiment Interface**: Load and query multi-modal neuroscience data (neural responses, eye tracking, treadmill, visual stimuli) through a single `Experiment` class
- **Flexible Interpolation**: Interpolate data at arbitrary time points with support for linear and nearest-neighbor methods
- **Multi-Session Support**: Combine data from multiple recording sessions into a single dataloader
- **Configurable Preprocessing**: YAML-based configuration for sampling rates, normalization, transforms, and filtering
- **PyTorch Integration**: Native PyTorch `Dataset` and `DataLoader` implementations optimized for training

## Docs
[![Docs](https://readthedocs.org/projects/experanto/badge/?version=latest)](https://experanto.readthedocs.io/)

## Installation

```bash
git clone https://github.com/sensorium-competition/experanto.git
cd experanto
pip install -e .
```

### Note

To replicate the `generate_sample` example, use the following command (see [allen_exporter](https://github.com/sensorium-competition/allen-exporter)):

```bash
pip install -e /path/to/allen_exporter
```

To replicate the `sensorium_example` (see [sensorium_2023](https://github.com/ecker-lab/sensorium_2023)), install neuralpredictors (see [neuralpredictors](https://github.com/sinzlab/neuralpredictors)) as well:

```bash
pip install -e /path/to/neuralpredictors
pip install -e /path/to/sensorium_2023
```

## Quick Start

### Loading an Experiment

```python
from experanto.experiment import Experiment

# Load a single experiment
exp = Experiment("/path/to/experiment")

# Query data at specific time points
import numpy as np
times = np.linspace(0, 10, 100)  # 100 time points over 10 seconds

# Get interpolated data from all devices
data = exp.interpolate(times)

# Or from a specific device
responses = exp.interpolate(times, device="responses")
```

### Configuration

Experanto uses YAML configuration files. See `configs/default.yaml` for all options:

```yaml
dataset:
  modality_config:
    responses:
      sampling_rate: 8
      chunk_size: 16
      transforms:
        normalization: "standardize"
    screen:
      sampling_rate: 30
      chunk_size: 60
      transforms:
        normalization: "normalize"

dataloader:
  batch_size: 16
  num_workers: 2
```
### Creating a DataLoader for Training

Here's a complete example showing how to create a PyTorch DataLoader for model training:
```python
from pathlib import Path
from experanto.experiment import Experiment
from experanto.datasets import ChunkDataset
from torch.utils.data import DataLoader

# Load experiments
experiment_paths = [
    "/path/to/experiment1",
    "/path/to/experiment2",
]
experiments = [Experiment(p) for p in experiment_paths]

# Create dataset
dataset = ChunkDataset(
    experiments=experiments,
    modalities=["responses", "screen"],  # neural responses + visual stimuli
    chunk_size=16,  # temporal window size
    sampling_rate=30,  # Hz
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

# Use in training loop
for batch in dataloader:
    responses = batch["responses"]  # shape: (batch, time, neurons)
    stimuli = batch["screen"]       # shape: (batch, time, height, width)
    # Your model training code here
```

For configuration options and advanced usage, see the [Configuration Options](https://experanto.readthedocs.io/en/latest/concepts/demo_configs.html) documentation.

## Documentation

Full documentation is available at [Read the Docs](https://experanto.readthedocs.io/).

- [Installation Guide](https://experanto.readthedocs.io/en/latest/concepts/installation.html)
- [Getting Started](https://experanto.readthedocs.io/en/latest/concepts/getting_started.html)
- [API Reference](https://experanto.readthedocs.io/en/latest/api.html)
- [Configuration Options](https://experanto.readthedocs.io/en/latest/concepts/demo_configs.html)

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and open an issue or submit a pull request on [GitHub](https://github.com/sensorium-competition/experanto).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Experanto

A Python package for interpolating recordings and stimuli in neuroscience experiments. Experanto provides a unified interface to load experimental data from multiple sessions and create efficient PyTorch dataloaders for machine learning applications.

## Features

- **Unified Experiment Interface**: Load and query multi-modal neuroscience data (neural responses, eye tracking, treadmill, visual stimuli) through a single `Experiment` class
- **Flexible Interpolation**: Interpolate data at arbitrary time points with support for linear and nearest-neighbor methods
- **Multi-Session Support**: Combine data from multiple recording sessions into a single dataloader
- **Configurable Preprocessing**: YAML-based configuration for sampling rates, normalization, transforms, and filtering
- **PyTorch Integration**: Native PyTorch `Dataset` and `DataLoader` implementations optimized for training

## Installation

```bash
git clone https://github.com/sensorium-competition/experanto.git
cd experanto
pip install -e .
```

### Optional Dependencies

For running the example notebooks:

```bash
# Allen Brain Observatory data export
pip install -e /path/to/allen_exporter
```
See [allen_exporter](https://github.com/sensorium-competition/allen-exporter)

## Quick Start

### Loading an Experiment

```python
from experanto import Experiment

# Load a single experiment
exp = Experiment("/path/to/experiment")

# Query data at specific time points
import numpy as np
times = np.linspace(0, 10, 100)  # 100 time points over 10 seconds

# Get interpolated data from all devices
data = exp.interpolate(times)

# Or from a specific device
responses = exp.interpolate(times, device_name="responses")
```

### Creating a DataLoader

```python
from experanto.dataloaders import get_multisession_dataloader

# Create a dataloader from multiple experiment paths
dataloader = get_multisession_dataloader(
    paths=["/path/to/exp1", "/path/to/exp2"],
    batch_size=32,
    config_path="configs/default.yaml"
)

for batch in dataloader:
    screen = batch["screen"]       # Visual stimuli
    responses = batch["responses"] # Neural responses
    # ... train your model
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

## Documentation

Full documentation is available at [Read the Docs](https://experanto.readthedocs.io/).

- [Installation Guide](https://experanto.readthedocs.io/en/latest/concepts/installation.html)
- [Getting Started](https://experanto.readthedocs.io/en/latest/concepts/getting_started.html)
- [API Reference](https://experanto.readthedocs.io/en/latest/api.html)
- [Configuration Options](https://experanto.readthedocs.io/en/latest/configuration.html)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/sensorium-competition/experanto).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use Experanto in your research, please cite:

```bibtex
@software{experanto,
  title = {Experanto: A Python Package for Neuroscience Data Interpolation},
  author = {Sinzlab},
  url = {https://github.com/sensorium-competition/experanto},
  year = {2025}
}
```

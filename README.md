# Experanto
Experanto is a Python package designed for interpolating recordings and stimuli in neuroscience experiments. It enables users to load single or multiple experiments and create efficient dataloaders for machine learning applications.

## Installation
To install Experanto, clone locally and run:
```bash
pip install -e /path_to/experanto
```

To replicate the `generate_sample` example, install:
```bash
pip install -e /path_to/allen_exporter
```
(Repository: [allen_exporter](https://github.com/sensorium-competition/allen-exporter))

To replicate the `sensorium_example`, also install the following with their dependencies:
```bash
pip install -e /path_to/neuralpredictors
```
(Repository: [neuralpredictors](https://github.com/sinzlab/neuralpredictors))

Additionally, clone the `sensorium_2023` repository and add the directory to the Python path using:
```python
import sys
sys.path.append('/path_to/sensorium_2023/')
```
(Repository: [sensorium_2023](https://github.com/ecker-lab/sensorium_2023))

Ensure you replace `/path_to/` with the actual path to the cloned repositories.

## Project structure

```
experanto/
├── experanto/
│   ├── __init__.py
│   ├── configs.py
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── experiment.py
│   ├── interpolators.py
│   ├── utils.py
├── configs/
│   ├── __init__.py
│   ├── default.yaml
├── tests/
│   ├── __init__.py
│   ├── create_mock_data.py
│   ├── test_sequence_interpolators.py
├── examples/                        
├── logs/                             
├── Dockerfile
├── docker-compose.yml
├── setup.py                    
├── requirements.txt                  
├── pytest.ini                  
└── README.md    
```
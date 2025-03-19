#### fast-forward pytorch dataloaders for neuroscience 
requirements
- required python >= 3.9
- use with latest nvidia NGC docker image (see dockerfile)
- pytorch >= 2.5.1


[demo notebook](./examples/demo.ipynb)

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
├── scripts/
│   ├── run_distributed_dataloaders.py
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

### Run distributed dataloader benchmark
- simply run the run_distributed_dataloaders.py script in ./scripts
- refer to the benchmarking.yaml to override default arguments

`torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29400 run_distributed_dataloaders.py datapath.root="/path/to/data/"`

- make sure to specify the directory where the example datasets sit in, i.e.:
  - `datapath.root="/path/to/data/"` 
- `dataloader.batch_size=64`
- `dataloader.num_workers=8`




Example output
```
[2025-03-14 01:34:36,495][__main__][INFO] - ===== Distributed Performance Summary =====
[2025-03-14 01:34:36,495][__main__][INFO] - Average throughput across all ranks: 22287.65 frames/second
[2025-03-14 01:34:36,495][__main__][INFO] - Min throughput: 20565.96, Max throughput: 26148.95
[2025-03-14 01:34:36,495][__main__][INFO] - Throughput imbalance: 25.05%
[2025-03-14 01:34:36,495][__main__][INFO] - Rank 0: Waiting for all processes to complete before cleanup...
[2025-03-14 01:34:37,497][__main__][INFO] - All processes completed. Starting cleanup...
[2025-03-14 01:34:37,867][__main__][INFO] - Rank 0: Successfully cleaned up process group
[2025-03-14 01:34:37,867][__main__][INFO] - Rank 0: Finished profiling and cleanup complete
[2025-03-14 01:34:37,984][__main__][INFO] - Rank 1: Successfully cleaned up process group
[2025-03-14 01:34:37,985][__main__][INFO] - Rank 1: Finished profiling and cleanup complete
[2025-03-14 01:34:38,162][__main__][INFO] - Rank 2: Successfully cleaned up process group
[2025-03-14 01:34:38,163][__main__][INFO] - Rank 2: Finished profiling and cleanup complete
[2025-03-14 01:34:38,368][__main__][INFO] - Rank 3: Successfully cleaned up process group
[2025-03-14 01:34:38,368][__main__][INFO] - Rank 3: Finished profiling and cleanup complete
```
# Experanto Codebase Analysis

This document provides a comprehensive analysis of the Experanto package structure, class relationships, potential issues, and documentation needs.

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Module Analysis](#module-analysis)
   - [intervals.py](#intervalspy)
   - [interpolators.py](#interpolatorspy)
   - [experiment.py](#experimentpy)
   - [datasets.py](#datasetspy)
   - [dataloaders.py](#dataloaderspy)
   - [configs.py](#configspy)
   - [utils.py](#utilspy)
   - [filters/](#filters)
4. [Class Hierarchy](#class-hierarchy)
5. [Data Flow](#data-flow)
6. [Potential Issues](#potential-issues)
7. [Docstring Requirements](#docstring-requirements)

---

## Package Overview

**Experanto** is a Python package for interpolating recordings and stimuli from neuroscience experiments. It enables loading single or multiple experiments and creating efficient dataloaders for machine learning applications.

### Core Concepts

1. **Experiment**: A folder containing multiple modalities (screen, responses, eye_tracker, treadmill)
2. **Modality**: A type of recorded data (visual stimuli, neural responses, behavioral data)
3. **Interpolation**: Resampling data at arbitrary time points
4. **Dataset**: PyTorch-compatible dataset that chunks and transforms experiment data
5. **Dataloader**: Handles batching and multi-session loading

### Supported Data Format

```
experiment/
  ├── screen/           # Visual stimuli (images/videos)
  │   ├── data/         # .npy or .mp4 files per trial
  │   ├── meta/         # .yml metadata per trial
  │   ├── meta.yml      # Global screen metadata
  │   └── timestamps.npy
  ├── responses/        # Neural recordings
  │   ├── data.mem      # Memory-mapped array (T x N_neurons)
  │   ├── meta.yml
  │   └── meta/
  │       ├── means.npy
  │       ├── stds.npy
  │       └── phase_shifts.npy  # Optional
  ├── eye_tracker/      # Same structure as responses
  └── treadmill/        # Same structure as responses
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER API                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  get_multisession_dataloader()    get_multisession_concat_dataloader()      │
│              │                                  │                            │
│              └──────────────┬───────────────────┘                            │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     DATALOADERS (dataloaders.py)                     │    │
│  │  - LongCycler: cycles through sessions until longest exhausted       │    │
│  │  - FastSessionDataLoader: optimized multi-session loader             │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DATASETS (datasets.py)                          │    │
│  │  - ChunkDataset: main dataset class for ML training                  │    │
│  │  - SimpleChunkedDataset: simpler version without filtering           │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     EXPERIMENT (experiment.py)                       │    │
│  │  - Experiment: loads and manages multiple device interpolators       │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   INTERPOLATORS (interpolators.py)                   │    │
│  │  ┌─────────────────┐    ┌─────────────────────────────────────┐      │    │
│  │  │   Interpolator  │    │         ScreenInterpolator          │      │    │
│  │  │     (ABC)       │    │  - Handles images/videos            │      │    │
│  │  └────────┬────────┘    │  - Uses ScreenTrial subclasses      │      │    │
│  │           │             └─────────────────────────────────────┘      │    │
│  │           ▼                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              SequenceInterpolator                            │    │    │
│  │  │  - Handles 1D time series (responses, eye_tracker, etc.)    │    │    │
│  │  │  - Supports linear and nearest_neighbor interpolation       │    │    │
│  │  └────────────────────────┬────────────────────────────────────┘    │    │
│  │                           │                                          │    │
│  │                           ▼                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │         PhaseShiftedSequenceInterpolator                     │    │    │
│  │  │  - For neurons with different phase shifts                   │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     INTERVALS (intervals.py)                         │    │
│  │  - TimeInterval: represents a time range [start, end]                │    │
│  │  - Operations: intersection, union, complement, uniquefy             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    SUPPORTING MODULES
┌─────────────────────────────────────────────────────────────────────────────┐
│  configs.py          │  utils.py                  │  filters/               │
│  - DEFAULT_CONFIG    │  - MultiEpochsDataLoader   │  - nan_filter           │
│  - Loads from YAML   │  - LongCycler/ShortCycler  │  - (placeholders for    │
│                      │  - SessionConcatDataset    │    gaze, responses,     │
│                      │  - FastSessionDataLoader   │    treadmill filters)   │
│                      │  - Various samplers        │                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Analysis

### intervals.py

**Purpose**: Provides time interval operations for determining valid data ranges.

#### Classes

| Class | Description | Needs Docstring |
|-------|-------------|-----------------|
| `TimeInterval` | NamedTuple representing a time range [start, end] | Yes - brief |

#### Functions

| Function | Description | Has Docstring | Quality |
|----------|-------------|---------------|---------|
| `uniquefy_interval_array` | Merges overlapping/adjacent intervals | Yes | Good (Google style) |
| `find_intersection_between_two_interval_arrays` | Finds intersection of two interval lists | No | Needs docstring |
| `find_intersection_across_arrays_of_intervals` | Finds common intersection across multiple lists | No | Needs docstring |
| `find_union_across_arrays_of_intervals` | Finds union across multiple interval lists | No | Needs docstring |
| `find_complement_of_interval_array` | Finds gaps not covered by intervals | Yes | Good (Google style) |
| `get_stats_for_valid_interval` | Calculates statistics about valid/invalid intervals | Yes | Good (Google style) |

**Potential Issues**: None identified.

---

### interpolators.py

**Purpose**: Handles resampling of data at arbitrary time points. This is the core interpolation engine.

#### Classes

| Class | Description | Needs Docstring |
|-------|-------------|-----------------|
| `Interpolator` | Abstract base class for all interpolators | Yes - class level |
| `SequenceInterpolator` | Interpolates 1D time series data | Yes - has minimal docstring |
| `PhaseShiftedSequenceInterpolator` | Handles per-neuron phase shifts | Yes |
| `ScreenInterpolator` | Interpolates visual stimuli (images/videos) | Yes - has minimal docstring |
| `ScreenTrial` | Base class for trial data | Yes |
| `ImageTrial` | Single-frame trials | Yes |
| `VideoTrial` | Multi-frame trials | Yes |
| `BlankTrial` | Blank/gray screen trials | Yes |
| `InvalidTrial` | Invalid trial placeholder | Yes |

#### Key Methods

| Method | Class | Description | Needs Docstring |
|--------|-------|-------------|-----------------|
| `create` | `Interpolator` | Factory method - creates appropriate interpolator | Yes |
| `interpolate` | All | Core method - returns data at given times | Yes |
| `valid_times` | `Interpolator` | Returns mask of valid time indices | Yes |
| `normalize_data` | `SequenceInterpolator` | Applies normalization | Yes |

**Potential Issues**:
1. `ScreenInterpolator.__init__` has type hint error: `tuple(int, int)` should be `Tuple[int, int]`
2. Magic number `1e-4` offset in `ScreenInterpolator.interpolate` should be documented
3. `SequenceInterpolator` has complex interpolation logic that needs math documentation

---

### experiment.py

**Purpose**: High-level interface for loading an experiment with multiple modalities.

#### Classes

| Class | Description | Needs Docstring |
|-------|-------------|-----------------|
| `Experiment` | Main class for loading experiments | Yes - has minimal docstring |

#### Key Methods

| Method | Description | Needs Docstring |
|--------|-------------|-----------------|
| `__init__` | Loads experiment from folder | Has docstring (needs formatting) |
| `_load_devices` | Internal: creates interpolators for each modality | Yes (private, low priority) |
| `interpolate` | Interpolates one or all devices at given times | Yes |
| `get_valid_range` | Gets valid time range for a device | Yes |
| `device_names` | Property: returns tuple of available devices | Yes |

**Potential Issues**:
1. `start_time` and `end_time` are overwritten in loop (only last device's times are kept)
2. Should probably compute intersection of all device time ranges

---

### datasets.py

**Purpose**: PyTorch Dataset classes for ML training.

#### Classes

| Class | Description | Needs Docstring |
|-------|-------------|-----------------|
| `SimpleChunkedDataset` | Basic dataset without filtering | Yes |
| `ChunkDataset` | Full-featured dataset with filtering, transforms | Yes - has long example docstring |

#### Key Methods (ChunkDataset)

| Method | Description | Has Docstring | Quality |
|--------|-------------|---------------|---------|
| `__init__` | Complex initialization with config | Yes | Needs reformatting |
| `_read_trials` | Reads trial metadata | No | Needs docstring |
| `initialize_statistics` | Loads normalization stats | Yes | Needs reformatting |
| `initialize_transforms` | Sets up data transforms | Yes | Minimal |
| `_get_callable_filter` | Instantiates filter functions | Yes | Partial (Google style) |
| `get_valid_intervals_from_filters` | Applies filters to get valid intervals | No | Needs docstring |
| `get_condition_mask_from_meta_conditions` | Creates boolean mask for trials | Yes | Good (Google style) |
| `get_screen_sample_mask_from_meta_conditions` | Creates mask for screen samples | Yes | Good (Google style) |
| `get_full_valid_sample_times` | Gets all valid starting points | Yes | Minimal |
| `shuffle_valid_screen_times` | Shuffles valid times for randomization | No | Needs docstring |
| `get_data_key_from_root_folder` | Extracts data key from path | Yes | Good (Google style) |
| `__getitem__` | Returns a data chunk | No | Needs docstring |
| `get_state` / `set_state` | State management for resumption | No | Needs docstring |

**Potential Issues**:
1. `DEFAULT_MODALITY_CONFIG = dict()` on line 32 shadows the import - likely a bug
2. `__init__` docstring contains YAML example that breaks Sphinx parsing
3. `get_data_key_from_root_folder` has undefined variable `path` (should be `root_folder`)
4. Missing `@classmethod` or `@staticmethod` decorator on `get_data_key_from_root_folder`

---

### dataloaders.py

**Purpose**: Functions for creating multi-session dataloaders.

#### Functions

| Function | Description | Has Docstring | Quality |
|----------|-------------|---------------|---------|
| `get_multisession_dataloader` | Creates dict of dataloaders per session | Yes | Poor formatting |
| `get_multisession_concat_dataloader` | Creates single concatenated dataloader | Yes | Partial |

**Potential Issues**:
1. Docstrings have formatting issues (inline `**kwargs` documentation)
2. Dataset name extraction logic is fragile (relies on path patterns)

---

### configs.py

**Purpose**: Loads default configuration from YAML.

#### Exports

| Name | Description | Needs Docstring |
|------|-------------|-----------------|
| `DEFAULT_CONFIG` | Full default configuration | Yes (module-level) |
| `DEFAULT_DATASET_CONFIG` | Dataset portion of config | Yes |
| `DEFAULT_MODALITY_CONFIG` | Per-modality configuration | Yes |
| `DEFAULT_DATALOADER_CONFIG` | Dataloader settings | Yes |

**Potential Issues**: None - simple config loading module.

---

### utils.py

**Purpose**: Utility classes for data loading and processing.

#### Functions

| Function | Description | Has Docstring | Quality |
|----------|-------------|---------------|---------|
| `replace_nan_with_batch_mean` | Replaces NaNs with column means | No | Needs docstring |
| `add_behavior_as_channels` | Adds behavior data as image channels | Yes | Poor formatting |
| `cycle` | Infinite iterator cycling | No | Needs docstring |

#### Classes

| Class | Description | Needs Docstring |
|-------|-------------|-----------------|
| `MultiEpochsDataLoader` | DataLoader that keeps workers across epochs | Yes - has brief comment |
| `Exhauster` | Steps through loaders sequentially | Yes - has brief docstring |
| `LongCycler` | Cycles until longest loader exhausted | Yes - has brief docstring |
| `ShortCycler` | Cycles until shortest loader exhausted | Yes - has brief docstring |
| `_RepeatSampler` | Internal: repeats sampler indefinitely | No (private) |
| `SessionConcatDataset` | Concatenates datasets with session tracking | Yes - brief |
| `SessionBatchSampler` | Batch sampler cycling through sessions | Yes - has docstring |
| `FastSessionDataLoader` | Optimized multi-session dataloader | Yes - has docstring |
| `SessionSpecificSampler` | Sampler for single session | Yes - has docstring |

**Potential Issues**:
1. `cycle` function shadows the imported `cycle` from itertools
2. Many classes have complex logic that would benefit from detailed docstrings
3. Debug `print` statements in `SessionConcatDataset` and `SessionBatchSampler`

---

### filters/

**Purpose**: Data quality filters that return valid time intervals.

#### Files

| File | Content | Status |
|------|---------|--------|
| `__init__.py` | Empty | OK |
| `common_filters.py` | `nan_filter` function | Needs docstring |
| `gaze_filters.py` | Empty | Placeholder |
| `responses_filters.py` | Empty | Placeholder |
| `treadmill_filters.py` | Empty | Placeholder |

**Note**: The filter architecture uses a factory pattern where functions return implementations.

---

## Class Hierarchy

```
Interpolator (ABC)
├── SequenceInterpolator
│   └── PhaseShiftedSequenceInterpolator
└── ScreenInterpolator

ScreenTrial
├── ImageTrial
├── VideoTrial
├── BlankTrial
└── InvalidTrial

torch.utils.data.Dataset
├── SimpleChunkedDataset
├── ChunkDataset
└── SessionConcatDataset

torch.utils.data.DataLoader
└── MultiEpochsDataLoader

torch.utils.data.Sampler
├── SessionBatchSampler
└── SessionSpecificSampler

# Not inheriting but important
FastSessionDataLoader  # Custom dataloader-like class
LongCycler            # Dataloader wrapper
ShortCycler           # Dataloader wrapper
Exhauster             # Dataloader wrapper
```

---

## Data Flow

```
1. LOADING
   User provides: root_folder path
                      │
                      ▼
   Experiment scans subfolders (screen/, responses/, etc.)
                      │
                      ▼
   For each modality folder:
      - Read meta.yml to determine type
      - Create appropriate Interpolator
      - Load timestamps and data

2. INTERPOLATION
   User provides: array of time points
                      │
                      ▼
   Interpolator.interpolate(times):
      - Filter to valid times
      - For SequenceInterpolator: index into data array
      - For ScreenInterpolator: find frames, load trial data
                      │
                      ▼
   Returns: (data_array, valid_mask)

3. DATASET ITERATION
   ChunkDataset.__getitem__(idx):
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
   Get start time              For each modality:
   from _valid_screen_times       - Calculate time array
                                  - Call interpolate()
                                  - Apply transforms
                      │
                      ▼
   Returns: dict with 'screen', 'responses', etc.

4. MULTI-SESSION LOADING
   get_multisession_dataloader(paths, config):
                      │
                      ▼
   For each path:
      - Create ChunkDataset
      - Wrap in MultiEpochsDataLoader
                      │
                      ▼
   Wrap all in LongCycler
                      │
                      ▼
   Iteration yields: (session_key, batch)
```

---

## Potential Issues

### Critical

| Location | Issue | Recommendation |
|----------|-------|----------------|
| `datasets.py:32` | `DEFAULT_MODALITY_CONFIG = dict()` shadows import | Remove this line |
| `datasets.py:596-598` | Undefined variable `path` in `get_data_key_from_root_folder` | Change to `root_folder` |
| `datasets.py:563` | Missing `@staticmethod` decorator | Add decorator |
| `interpolators.py:323` | Type hint `tuple(int, int)` is invalid | Change to `Tuple[int, int]` |

### Medium

| Location | Issue | Recommendation |
|----------|-------|----------------|
| `experiment.py:54-55` | Only last device's time range is kept | Compute intersection |
| `utils.py:127` | `cycle` shadows itertools import | Rename to `infinite_cycle` |
| `utils.py:234,326,350,484` | Debug print statements | Remove or use logging |
| `interpolators.py:433` | Magic offset `1e-4` | Document why this is needed |

### Low

| Location | Issue | Recommendation |
|----------|-------|----------------|
| Various | Inconsistent docstring styles | Standardize to NumPy style |
| `dataloaders.py:50-55` | Fragile path parsing | Use metadata file instead |

---

## Docstring Requirements

### Priority 1: Public API (Must Have)

These are the main entry points users interact with:

| Module | Item | Current State |
|--------|------|---------------|
| `experiment.py` | `Experiment` class | Minimal |
| `experiment.py` | `Experiment.interpolate` | None |
| `datasets.py` | `ChunkDataset` class | Has example, needs reformatting |
| `datasets.py` | `ChunkDataset.__getitem__` | None |
| `dataloaders.py` | `get_multisession_dataloader` | Poor formatting |
| `dataloaders.py` | `get_multisession_concat_dataloader` | Poor formatting |
| `configs.py` | Module docstring | None |

### Priority 2: Core Classes (Should Have)

| Module | Item | Current State |
|--------|------|---------------|
| `interpolators.py` | `Interpolator` class | None |
| `interpolators.py` | `SequenceInterpolator` class | Minimal |
| `interpolators.py` | `ScreenInterpolator` class | Minimal |
| `interpolators.py` | `Interpolator.interpolate` (abstract) | Minimal |
| `intervals.py` | `TimeInterval` class | None |
| `utils.py` | `LongCycler` class | Minimal |
| `utils.py` | `FastSessionDataLoader` class | Good |

### Priority 3: Supporting Classes (Nice to Have)

| Module | Item | Current State |
|--------|------|---------------|
| `interpolators.py` | `PhaseShiftedSequenceInterpolator` | None |
| `interpolators.py` | `ScreenTrial` and subclasses | Minimal |
| `datasets.py` | `SimpleChunkedDataset` | None |
| `utils.py` | `SessionConcatDataset` | Minimal |
| `utils.py` | `MultiEpochsDataLoader` | Comment only |
| `filters/common_filters.py` | `nan_filter` | None |

### Priority 4: Internal/Private (Optional)

| Module | Item | Current State |
|--------|------|---------------|
| `intervals.py` | Helper functions | Some have docstrings |
| `datasets.py` | Private methods (`_read_trials`, etc.) | None |
| `utils.py` | Samplers and internal classes | Some have docstrings |

---

## Recommended Docstring Examples

### For Interpolator.interpolate (with math)

```python
def interpolate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate data values at specified time points.

    Parameters
    ----------
    times : numpy.ndarray
        1D array of time points (in seconds) at which to interpolate.
        Times outside the valid interval will be excluded.

    Returns
    -------
    data : numpy.ndarray
        Interpolated values with shape ``(n_valid_times, n_features)``.
    valid : numpy.ndarray
        Boolean mask indicating which input times were valid.

    Notes
    -----
    For linear interpolation, values are computed as:

    .. math::

        y(t) = y_0 + (y_1 - y_0) \\frac{t - t_0}{t_1 - t_0}

    where :math:`t_0` and :math:`t_1` are the surrounding sample times.

    Examples
    --------
    >>> interp = SequenceInterpolator('/path/to/data')
    >>> times = np.array([1.0, 1.5, 2.0])
    >>> data, valid = interp.interpolate(times)
    >>> data.shape
    (3, 100)  # 3 times, 100 neurons
    """
```

### For ChunkDataset

```python
class ChunkDataset(Dataset):
    """PyTorch Dataset for chunked neuroscience experiment data.

    This dataset loads experiment data and provides temporally-chunked
    samples suitable for training neural networks with temporal context
    (e.g., 3D convolutions, RNNs).

    Parameters
    ----------
    root_folder : str
        Path to the experiment directory containing modality subfolders.
    global_sampling_rate : float, optional
        Sampling rate in Hz applied to all modalities. If None, uses
        per-modality rates from config.
    global_chunk_size : int, optional
        Number of samples per chunk. If None, uses per-modality sizes.
    modality_config : dict
        Configuration for each modality including transforms, filters,
        and interpolation settings.
    cache_data : bool, default=False
        If True, loads all data into memory for faster access.
    seed : int, optional
        Random seed for reproducible shuffling.

    Attributes
    ----------
    device_names : tuple
        Names of available modalities (e.g., 'screen', 'responses').
    data_key : str
        Unique identifier for this dataset.

    See Also
    --------
    Experiment : Lower-level interface for data access.
    get_multisession_dataloader : Load multiple datasets.

    Examples
    --------
    >>> from experanto.datasets import ChunkDataset
    >>> from experanto.configs import DEFAULT_MODALITY_CONFIG
    >>> dataset = ChunkDataset(
    ...     '/path/to/experiment',
    ...     global_sampling_rate=30,
    ...     global_chunk_size=60,
    ...     modality_config=DEFAULT_MODALITY_CONFIG
    ... )
    >>> sample = dataset[0]
    >>> sample['screen'].shape
    torch.Size([1, 60, 144, 256])
    """
```

---

## Next Steps

1. **Fix critical bugs** before adding docstrings
2. **Add module-level docstrings** to each file
3. **Document Priority 1 items** (public API)
4. **Document Priority 2 items** (core classes)
5. **Rerun Sphinx build** to verify formatting
6. **Add math notation** where interpolation formulas are relevant

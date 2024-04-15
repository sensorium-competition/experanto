# Experanto
Python package to interpolate recordings and stimuli of neuroscience experiments 

## Use specification
- Instantiation
```python
dat = Experiment('dataset-folder', discretization=30) # 30Hz
```

- Single frame or sequence access
```python
item = dat[10]
sequence = dat[10:100]
```


## Data Folder Structure

```
dataset-folder/
  images/
    meta.json # what type of interpolator should be used.
    0001/
      meta.json
      timestamps.npz
      meta/
        condition_hash.npy
        trial_idx.npy
      data/
        img01.png
        img02.png
        ...
    0002/
      ...
  videos/
    meta.json
    0001/
      meta.json
      timestamps.npz
      meta/
        condition_hash.npy
        trial_idx.npy
      data/
        img01.png
        img02.png
        ...
    0002/
      ...
  pupil_dilation/
    meta.json
    timestamps.npz
  running/
    meta.json
    timestamps.npz
  multiunit/
    meta.json
    timestamps.npz
  poses/
    meta.json
    timestamps.npz
```

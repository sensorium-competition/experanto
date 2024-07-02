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
  screen/
    meta.json # what type of interpolator should be used for which block / which data type each block is
    0001/ # this could be a block of images
      meta.json # what is actually here? should it be deleted?
      timestamps.npz
      meta/
        condition_hash.npy
        trial_idx.npy
      data/
        img01.png
        img02.png
        ...
    0002/ # this could be a block of videos
      ...
    0003/ # this could be a abother block of images 
      ...
  eye_tracker/
    meta.json
    timestamps.npz
  running_wheel/
    meta.json
    timestamps.npz
  multiunit/
    meta.json
    timestamps.npz
  poses/
    meta.json
    timestamps.npz
```

## Example for meta.json

```
modality: images
```
or 
```
0001: images
0002: videos
0003: images
...
```

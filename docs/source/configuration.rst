Configuration
=============

Experanto uses YAML configuration files to control dataset and dataloader behavior.
The default configuration is loaded from ``configs/default.yaml``.

.. automodule:: experanto.configs
   :members:
   :undoc-members:

Default Configuration
---------------------

Below is the default configuration file (``configs/default.yaml``):

.. code-block:: yaml

   dataset:
     global_sampling_rate: null
     global_chunk_size: null
     add_behavior_as_channels: False
     replace_nans_with_means: False
     cache_data: False
     out_keys: ["screen", "responses", "eye_tracker", "treadmill", "timestamps"]
     normalize_timestamps: True
     modality_config:
       screen:
         keep_nans : False
         sampling_rate: 30
         chunk_size: 60
         valid_condition:
           tier: "train"
         offset: 0
         sample_stride: 1
         include_blanks: True
         transforms:
           normalization: "normalize"
           Resize:
             _target_: "torchvision.transforms.v2.Resize"
             size: [144, 256]
         interpolation:
           rescale: True
           rescale_size: [144, 256]
       responses:
         keep_nans : False
         sampling_rate: 8
         chunk_size: 16
         offset: 0.0 # in seconds
         transforms:
           normalization: "standardize"
         interpolation:
           interpolation_mode: "nearest_neighbor"
         filters:
           nan_filter:
             __target__: experanto.filters.common_filters.nan_filter
             __partial__: True
             vicinity: 0.05
       eye_tracker:
         keep_nans : False
         sampling_rate: 30
         chunk_size: 60
         offset: 0
         transforms:
           normalization: "normalize"
         interpolation:
           interpolation_mode: "nearest_neighbor"
         filters:
           nan_filter:
             __target__: experanto.filters.common_filters.nan_filter
             __partial__: True
             vicinity: 0.05
       treadmill:
         keep_nans : False
         sampling_rate: 30
         chunk_size: 60
         offset: 0
         transforms:
           normalization: "normalize"
         interpolation:
           interpolation_mode: "nearest_neighbor"
         filters:
           nan_filter:
             __target__: experanto.filters.common_filters.nan_filter
             __partial__: True
             vicinity: 0.05

   dataloader:
     batch_size: 16
     shuffle: true
     num_workers: 2
     pin_memory: True
     drop_last: True
     prefetch_factor: 2

Configuration Options
---------------------

Dataset Options
^^^^^^^^^^^^^^^

``global_sampling_rate``
   Override sampling rate for all modalities. Set to ``null`` to use per-modality rates.

``global_chunk_size``
   Override chunk size for all modalities. Set to ``null`` to use per-modality sizes.

``add_behavior_as_channels``
   If ``True``, concatenate behavioral data (eye tracker, treadmill) as additional
   channels to the screen data.

``replace_nans_with_means``
   If ``True``, replace NaN values with the mean of non-NaN values.

``cache_data``
   If ``True``, cache interpolated data in memory for faster access.

``out_keys``
   List of modality keys to include in the output dictionary.

``normalize_timestamps``
   If ``True``, normalize timestamps to start from 0.

Modality Options
^^^^^^^^^^^^^^^^

Each modality (screen, responses, eye_tracker, treadmill) supports:

``keep_nans``
   Whether to keep NaN values in the output.

``sampling_rate``
   Sampling rate in Hz for this modality.

``chunk_size``
   Number of samples per chunk.

``offset``
   Time offset in seconds relative to the screen timestamps.

``transforms``
   Dictionary of transforms to apply. Supports ``"normalize"`` (0-1 scaling)
   and ``"standardize"`` (z-score normalization).

``interpolation``
   Interpolation settings including ``interpolation_mode`` (``"linear"`` or
   ``"nearest_neighbor"``).

``filters``
   Dictionary of filter functions to apply to the data.

Dataloader Options
^^^^^^^^^^^^^^^^^^

``batch_size``
   Number of samples per batch.

``shuffle``
   Whether to shuffle samples.

``num_workers``
   Number of worker processes for data loading.

``pin_memory``
   If ``True``, pin memory for faster GPU transfer.

``drop_last``
   If ``True``, drop the last incomplete batch.

``prefetch_factor``
   Number of batches to prefetch per worker.

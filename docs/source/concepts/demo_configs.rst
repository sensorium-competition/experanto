Dataset and dataloader configuration
====================================

This section describes the dataset configuration used to load experiments into the Experanto dataloaders. It includes global settings, modality-specific configurations, and dataloader parameters.

Default YAML configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    dataset:
      global_sampling_rate: null
      global_chunk_size: null
      add_behavior_as_channels: false
      replace_nans_with_means: false
      cache_data: false
      out_keys:
        - screen
        - responses
        - eye_tracker
        - treadmill
      modality_config:
        screen:
          keep_nans: false
          sampling_rate: 30
          chunk_size: 60
          valid_condition:
            tier: train
          offset: 0
          sample_stride: 1
          include_blanks: true
          transforms:
            normalization: normalize
            Resize:
              _target_: torchvision.transforms.v2.Resize
              size:
                - 144
                - 256
          interpolation:
            rescale: true
            rescale_size:
              - 144
              - 256
        responses:
          keep_nans: false
          sampling_rate: 8
          chunk_size: 16
          offset: 0.0
          transforms:
            normalization: standardize
          interpolation:
            interpolation_mode: nearest_neighbor
          filters:
            nan_filter:
              __target__: experanto.filters.common_filters.nan_filter
              __partial__: true
              vicinity: 0.05
        eye_tracker:
          keep_nans: false
          sampling_rate: 30
          chunk_size: 60
          offset: 0
          transforms:
            normalization: normalize
          interpolation:
            interpolation_mode: nearest_neighbor
          filters:
            nan_filter:
              __target__: experanto.filters.common_filters.nan_filter
              __partial__: true
              vicinity: 0.05
        treadmill:
          keep_nans: false
          sampling_rate: 30
          chunk_size: 60
          offset: 0
          transforms:
            normalization: normalize
          interpolation:
            interpolation_mode: nearest_neighbor
          filters:
            nan_filter:
              __target__: experanto.filters.common_filters.nan_filter
              __partial__: true
              vicinity: 0.05

    dataloader:
      batch_size: 16
      shuffle: true
      num_workers: 2
      pin_memory: true
      drop_last: true
      prefetch_factor: 2


Viewing the configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from omegaconf import OmegaConf, open_dict
  from experanto.configs import DEFAULT_CONFIG as cfg

  print(OmegaConf.to_yaml(cfg))


Modifying the configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can change parameters programmatically:

.. code-block:: python

    cfg.dataset.modality_config.screen.include_blanks = True
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg.dataloader.num_workers = 8


Configuration options
^^^^^^^^^^^^^^^^^^^^^

Dataset options
"""""""""""""""

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

Modality options
""""""""""""""""

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

Dataloader options
""""""""""""""""""

All standard ``torch.utils.data.DataLoader`` options are supported. See the
`PyTorch DataLoader documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
for the full list of available parameters.

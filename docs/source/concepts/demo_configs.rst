Dataset and Dataloader Configuration
====================================

This section describes the dataset configuration used to load experiments into the Experanto dataloaders. It includes global settings, modality-specific configurations, and dataloader parameters.

Default YAML Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from omegaconf import OmegaConf, open_dict
  from experanto.configs import DEFAULT_CONFIG as cfg

  print(OmegaConf.to_yaml(cfg))

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

Modifying the Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can change parameters programmatically:

.. code-block:: python

    cfg.dataset.modality_config.screen.include_blanks = True
    cfg.dataset.modality_config.screen.valid_condition = {"tier": "train"}
    cfg.dataloader.num_workers = 8

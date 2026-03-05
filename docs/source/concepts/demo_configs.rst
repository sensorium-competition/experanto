.. _dataset_and_dataloader_configuration:

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
            normalization: normalize_variance_only
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
   Override sampling rate for all modalities. Set to ``None`` to use
   per-modality rates.

``global_chunk_size``
   Override chunk size (number of time steps/data points) for all modalities.
   Set to ``None`` to use per-modality sizes.

   The time window covered by a chunk is ``chunk_size / sampling_rate``, so
   the ``global_sampling_rate`` should be taken into account:

   - **With** ``global_sampling_rate`` set: all modalities share the same
     output rate, so a single ``global_chunk_size`` unambiguously gives every
     modality the same time window.
   - **Without** ``global_sampling_rate`` (per-modality rates active):
     different modalities have different rates, so the same sample count
     produces different durations. In this case, leave ``global_chunk_size``
     as ``None`` and set ``chunk_size`` per modality instead.

``add_behavior_as_channels``
   If ``True``, concatenate behavioral data (e.g., eye tracker, treadmill) as
   additional channels to the screen data.

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

Each modality (e.g., screen, responses, eye_tracker, treadmill) supports:

``keep_nans``
   Whether to keep NaN values in the output.

``sampling_rate``
   Controls the spacing of the time points that the dataset constructs and
   passes to :meth:`~experanto.experiment.Experiment.interpolate`. Concretely,
   each item in the dataset requests values at times
   ``start, start + 1/sampling_rate, start + 2/sampling_rate, …``. The
   interpolator then interpolates the stored raw samples at those points.

``chunk_size``
   Number of **time steps/data points** returned per item for this modality.
   Internally, ``sampling_rate`` defines the spacing of the time points passed
   to the interpolator, so the covered time window is:

   .. math::

      \text{duration (s)} = \frac{\text{chunk\_size}}{\text{sampling\_rate}}

   Note that ``sampling_rate`` here controls the *spacing* of the time points
   requested from the underlying experiment (see ``sampling_rate`` above). The
   native acquisition rate of the signal does not matter (the interpolator simply looks up the stored values closest to each requested time, e.g.).

   When per-modality output rates differ, ``chunk_size`` must be set per
   modality to cover the same time window. The default configuration keeps
   all modalities at a 2-second window while using different output rates:

   ============  =============  ===========  ===========
   Modality      sampling_rate  chunk_size   Duration
   ============  =============  ===========  ===========
   screen        30 Hz          60           2 s
   eye_tracker   30 Hz          60           2 s
   treadmill     30 Hz          60           2 s
   responses     8 Hz           16           2 s
   ============  =============  ===========  ===========

   If you unify all rates with ``global_sampling_rate``, use
   ``global_chunk_size`` instead and this per-modality value is ignored.
   In general: ``chunk_size = desired_duration_seconds * sampling_rate``.

``offset``
   Time offset in seconds applied to the time points constructed for this
   modality. For example, if the screen is queried at times
   ``[t, t + 1/sampling_rate, …]``, setting ``offset = 0.1`` on responses
   means responses are queried at ``[t + 0.1, t + 0.1 + 1/sampling_rate, …]``.
   Useful for aligning modalities with known temporal delays relative to the
   screen stimulus.

``transforms``
   Dictionary of transforms to apply at the dataset level. This is modality
   specific, i.e., not all modalities support the same set of transforms. Some
   examples include ``"normalize"`` for sequences, such as eye_tracker,
   and ``"standardize"`` for responses.

   To understand how transforms are loaded and applied internally, refer to
   :meth:`experanto.datasets.ChunkDataset.initialize_transforms`. If you need
   to implement a custom transform, we recommend following the same pattern
   used there. In particular, note how each entry in the ``transforms``
   dictionary is checked and, when it is a config ``dict``, instantiated via
   Hydra before being added to the transform pipeline.

   You can point Experanto to any callable (function or class) by using
   Hydra's ``_target_`` key, which triggers
   `hydra.utils.instantiate <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_
   under the hood (e.g., ``_target_: my_package.my_module.MyTransform``).

``interpolation``
   Interpolation settings. This is modality specific, i.e., not all modalities
   support the same set of interpolation methods. Some examples include
   ``"rescale"`` for the screen and ``"interpolation_mode"`` (e.g.,
   ``"nearest_neighbor"``) for sequences.

``filters``
   Dictionary of filter functions to apply to the data.

Dataloader options
""""""""""""""""""

All standard ``torch.utils.data.DataLoader`` options are supported. See the
`PyTorch DataLoader documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
for the full list of available parameters.

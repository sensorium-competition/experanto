Classes & Functions
===================

This section documents all public classes and functions in experanto.

Core Classes
------------

.. autosummary::
   :toctree: generated
   :template: custom-class-template.rst
   :nosignatures:

   experanto.experiment.Experiment
   experanto.datasets.ChunkDataset

Interpolators
-------------

.. autosummary::
   :toctree: generated
   :template: custom-class-template.rst
   :nosignatures:

   experanto.interpolators.Interpolator
   experanto.interpolators.SequenceInterpolator
   experanto.interpolators.PhaseShiftedSequenceInterpolator
   experanto.interpolators.ScreenInterpolator
   experanto.interpolators.TimeIntervalInterpolator
   experanto.interpolators.ScreenTrial
   experanto.interpolators.ImageTrial
   experanto.interpolators.VideoTrial
   experanto.interpolators.BlankTrial
   experanto.interpolators.InvalidTrial

Time Intervals
--------------

.. autosummary::
   :toctree: generated
   :template: custom-class-template.rst
   :nosignatures:

   experanto.intervals.TimeInterval

.. autosummary::
   :toctree: generated
   :nosignatures:

   experanto.intervals.uniquefy_interval_array
   experanto.intervals.find_intersection_between_two_interval_arrays
   experanto.intervals.find_intersection_across_arrays_of_intervals
   experanto.intervals.find_union_across_arrays_of_intervals
   experanto.intervals.find_complement_of_interval_array
   experanto.intervals.get_stats_for_valid_interval

Dataloaders
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   experanto.dataloaders.get_multisession_dataloader
   experanto.dataloaders.get_multisession_concat_dataloader

Utilities
---------

.. autosummary::
   :toctree: generated
   :template: custom-class-template.rst
   :nosignatures:

   experanto.utils.LongCycler
   experanto.utils.ShortCycler
   experanto.utils.FastSessionDataLoader
   experanto.utils.MultiEpochsDataLoader
   experanto.utils.SessionConcatDataset
   experanto.utils.SessionBatchSampler
   experanto.utils.SessionSpecificSampler

.. autosummary::
   :toctree: generated
   :nosignatures:

   experanto.utils.add_behavior_as_channels

Filters
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   experanto.filters.common_filters.nan_filter

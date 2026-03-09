.. _loading_single_experiment:

Loading a single experiment
===========================

To load an experiment, we use the :class:`~experanto.experiment.Experiment`
class. This class aggregates all modalities and their respective interpolators
in a single object. Its main job is to unify the access to all modalities.

:meth:`Experiment.interpolate <experanto.experiment.Experiment.interpolate>` accepts an arbitrary array of time
points and returns the corresponding values for each modality by looking them
up in the raw stored data (e.g., using nearest-neighbour or linear
interpolation for sequences). When you need a regular sampling grid or
fixed-length intervals (chunks) for training, a dataset object such as
:class:`~experanto.datasets.ChunkDataset` should be used **on top of**
:class:`~experanto.experiment.Experiment`. There you can define the time
discretization (via ``sampling_rate`` and ``chunk_size``), construct the
appropriate time points, and delegate the data retrieval to the
underlying :class:`~experanto.experiment.Experiment`.

Loading an experiment
---------------------
A single experiment can be loaded using the following approach:

.. code-block:: python

    # Import the Experiment class from Experanto
    from experanto.experiment import Experiment

    # Set the experiment folder as the root directory
    root_folder = '../data/allen_data/experiment_951980471_train'

    # Initialize the Experiment object
    e = Experiment(root_folder)

Checking available modalities
-----------------------------
All compatible modalities for the loaded experiment can be checked using:

.. code-block:: python

    print("Available experiment devices:", e.devices.keys())

.. code-block:: text

    Available experiment devices: ['eye_tracker', 'screen', 'treadmill', 'responses']

Interpolating data
------------------
Once the modalities are identified, we can interpolate their data using
:meth:`~experanto.experiment.Experiment.interpolate`.

:meth:`~experanto.experiment.Experiment.interpolate` accepts any 1-D array of
time points and returns a NumPy array containing only the **valid** time
points — those that fall within the recorded range of the requested modality.
The returned array therefore has length ``n_valid``, which is less than or
equal to ``len(times)``, and its shape is modality-dependent:

* **Sequence modalities** (``responses``, ``eye_tracker``, ``treadmill``):
  ``(n_valid, n_signals)``
* **Screen modality** (``screen``): ``(n_valid, H, W)`` for grayscale stimuli,
  or ``(n_valid, H, W, C)`` when colour channels are present.

Pass ``return_valid=True`` to also receive ``valid``, an **integer index
array** into the original ``times`` array, such that ``times[valid]`` gives
the time points that correspond row-for-row to the returned data array.

When you call :meth:`~experanto.experiment.Experiment.interpolate` **without**
a ``device`` argument, every modality receives the *same* time array. Because
each modality may have a different valid range, the ``n_valid`` count can
differ between modalities. If you need different time densities per modality,
call :meth:`~experanto.experiment.Experiment.interpolate` separately with a
different ``times`` array for each ``device``. This is exactly what
:class:`~experanto.datasets.ChunkDataset` does internally.

The following example interpolates a 20-second window at 2 time points per
second. Any requested time that falls outside a valid screen trial is
excluded, so ``video.shape[0]`` may be less than ``len(times)``:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # Define the time window for interpolation
    times = np.arange(4300., 4320., 0.5)

    # Retrieve the frames as a NumPy float32 array with shape (n_valid, H, W)
    # ``valid`` is an integer index array into ``times``; times[valid] gives
    # the time point for each row of ``video`` (screen is blank outside trials).
    video, valid = e.interpolate(times, device="screen", return_valid=True)

    # ``video`` already contains only the n_valid frames; no further indexing needed.
    video_np = video.astype(np.uint8)
    n_frames, height, width = video_np.shape

    # Create a figure and axis for animation
    fig, ax = plt.subplots()
    img = ax.imshow(video_np[0], cmap='gray', vmin=0, vmax=255)

    def update(frame):
        img.set_array(video_np[frame])
        ax.set_title(f'Time step: {frame}')
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

    plt.close(fig)
    HTML(ani.to_jshtml())

This should show a speed up version of the video from the sample experiment.

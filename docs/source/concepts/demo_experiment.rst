.. _loading_single_experiment:

Loading a single experiment
===========================

To load an experiment, we use the :class:`~experanto.experiment.Experiment`
class. This class aggregates all modalities and their respective interpolators
in a single object. Its main job is to unify the access to all modalities.

:class:`~experanto.experiment.Experiment` accepts an arbitrary array of time
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
time points and returns, for each modality, an array of shape
``(len(times), n_signals)``. The number of returned points is always
``len(times)``, regardless of the native acquisition rate of the modality.

When you call :meth:`~experanto.experiment.Experiment.interpolate` **without**
a ``device`` argument, every modality receives the *same* time array. This
means modalities with low native rates can return repeated values for
consecutive requested times that fall in the same native sample (nearest
neighbour), while modalities with high native rates will effectively be
sub-sampled (the behavior is interpolator-dependent). If you need different
time densities per modality, call :meth:`~experanto.experiment.Experiment.interpolate`
separately with a different ``times`` array for each ``device``. This is
exactly what :class:`~experanto.datasets.ChunkDataset` does internally.

The following example interpolates a 20-second window at 2 time points per
second, resulting in 40 screen frames:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # Define the time window for interpolation
    times = np.arange(4300., 4320., 0.5)

    # Retrieve the frames as a torch array with shape (C, T, H, W)
    video, valid = e.interpolate(times, device="screen")

    # Convert to NumPy and prepare for visualization
    video_np = video.numpy().astype(int)
    channels, n_frames, height, width = video_np.shape
    video_np = np.transpose(video_np, (1, 2, 3, 0))

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

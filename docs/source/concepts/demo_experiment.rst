.. _loading_single_experiment:

Loading a single experiment
===========================

To load an experiment, we use the **Experiment** class. This is particularly useful for testing whether the formatting and interpolation behave as expected before loading multiple experiments into dataset objects.

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
Once the modalities are identified, we can interpolate their data. 
The following example interpolates a 20-second window with 2 frames per second, resulting in 40 images:

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

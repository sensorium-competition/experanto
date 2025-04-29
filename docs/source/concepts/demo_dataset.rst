
.. _loading_dataset:

Loading a dataset object
========================

Dataset objects organize experimental data (from **Experiment** class) for machine learning tasks, offering project-specific and configurable access for training and evaluation. They often serve as a source for creating **dataloaders**.

Key features of dataset objects
-------------------------------

Dataset objects provide several essential features:

- **Sampling Rate**: Defines the frequency of equally spaced interpolation times across the entire experiment. This ensures consistency in temporal data alignment.
- **Chunk Size**: Determines the number of values returned when calling the ``__getitem__`` method. This is crucial, for example, for deep learning models that use 3D convolutions over time, where single elements or small chunk sizes are insufficient to capture meaningful temporal patterns.
- **Modality Configuration**: Specifies the details of the interpolation, including:

  - The **interpolation method** used.
  - **Conditions** that the data must fulfill.
  - **Transformations** applied to the data (e.g., normalization, resizing, cropping, greyscale conversion).

Loading a dataset
-----------------
To load a dataset, follow the steps below:

.. code-block:: python

    from experanto.datasets import ChunkDataset
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from experanto.configs import DEFAULT_CONFIG as cfg

    cfg.dataset.modality_config.screen.transforms.Resize.size = [144, 144] 
    cfg.dataset.modality_config.screen.interpolation.rescale_size = [144, 144]
    cfg.dataset.modality_config.screen.transforms.greyscale = True
    modality_cfg = cfg.dataset.modality_config

    # Extract only 'screen' and 'responses' or other modalities if necessecary for single session loading
    selected_modalities = OmegaConf.create({
        'screen': modality_cfg.screen,
        'responses': modality_cfg.responses
    })

    root_folder = '../data/allen_data'
    sampling_rate = 60
    chunk_size = 60 # since we also use video data we always use chunks of images to also consider temporal developements

    train_dataset = ChunkDataset(root_folder=f'{root_folder}/experiment_951980471_train', global_sampling_rate=sampling_rate,
            global_chunk_size=chunk_size, modality_config = selected_modalities)

This configuration ensures that:

- **Screen data** is preprocessed with normalization (per default), resizing, and greyscale conversion.
- **Response data** undergoes standardization and nearest-neighbor interpolation (per default).

Other modalities can be defined in the same manner as **Responses**. If your desired modalities do not match our existing data structures and config layout, you will need to implement them yourself.
We appreciate contributions to Experanto in the form of pull requests via GitHub to make more modalities accessible.

Sampling data from the dataset
------------------------------
We can confirm the creation and functionality of our datasets by sampling some data.
To sample data from the dataset, we can simply index into it. For example, to sample the first data chunk:

.. code-block:: python

    # Interpolation showcase using the dataset object
    sample = train_dataset[0]

    # Print the keys and their respective shapes
    print(sample.keys())
    for key in sample.keys():
        print(f'This is shape {sample[key].shape} for modality {key}')

This will output something like:

.. code-block:: text

    dict_keys(['screen', 'responses'])
    This is shape torch.Size([1, 60, 144, 144]) for modality screen
    This is shape torch.Size([60, 12]) for modality responses

Defining dataloaders
---------------------
Once the dataset is verified, we can define **DataLoader** objects for training or other purposes. This allows easy batch processing during training:

.. code-block:: python

    # Define a DataLoader for the training set
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

Verifying dataloader functionality
----------------------------------
To confirm that the **DataLoader** works as expected, we can iterate over it and inspect the batch data. For example, to check the shapes of the data in each batch:

.. code-block:: python

    # Interpolation showcase using the data_loaders
    for batch_idx, batch_data in enumerate(train_data_loader):
        # batch_data is a dictionary with keys 'screen', 'responses'
        screen_data = batch_data['screen']
        responses_data = batch_data['responses']
        
        # Print or inspect the batch
        print(f"Batch {batch_idx}:")
        print("Screen Data:", screen_data.shape)
        print("Responses:", responses_data.shape)
        break

This will output something like:

.. code-block:: text

    Batch 0:
    Screen Data: torch.Size([32, 1, 60, 144, 144])
    Responses: torch.Size([32, 60, 12])

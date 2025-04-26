
.. _loading_dataset:

Loading a Dataset Object
========================

Dataset objects are the core of the Experanto library. They extend single experiments by adding functionality that allows them to be used directly as **dataloaders** for machine learning tasks.

Key Features of Dataset Objects
-------------------------------
Dataset objects provide several essential features:

- **Sampling Rate**: Defines the frequency of equally spaced interpolation times across the entire experiment. This ensures consistency in temporal data alignment.
- **Chunk Size**: Determines the number of values returned when calling the ``__getitem__`` method. This is crucial for deep learning models utilizing **3D convolutions over time**, as single elements or small chunk sizes would be insufficient for meaningful temporal patterns.
- **Modality Configuration**: Specifies the details of the interpolation, including:

  - The **interpolation method** used.
  - **Conditions** that the data must fulfill.
  - **Transforms** applied to the data (e.g., normalization, resizing, cropping, greyscale conversion).

Loading a Dataset
-----------------
To load a dataset, follow the steps below:

.. code-block:: python

    import sys
    from experanto.datasets import ChunkDataset
    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    from omegaconf import OmegaConf, open_dict
    from experanto.configs import DEFAULT_CONFIG as cfg

    cfg.dataset.modality_config.screen.transforms.Resize.size = [144,144] 
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

- **Screen data** is preprocessed with normalization, resizing, cropping, and greyscale conversion.
- **Response data** undergoes standardization and nearest-neighbor interpolation.

Other modalities can be defined in the same manner as **Responses**.

Sampling Data from the Dataset
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
    This is shape torch.Size([1, 32, 144, 144]) for modality screen
    This is shape torch.Size([32, 12]) for modality responses

Defining DataLoaders
---------------------
Once the dataset is verified, we can define **DataLoader** objects for training or other purposes. This allows easy batch processing during training:

.. code-block:: python

    # Define a DataLoader for the training set
    data_loader['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True)

Verifying DataLoader Functionality
----------------------------------
To confirm that the **DataLoader** works as expected, we can iterate over it and inspect the batch data. For example, to check the shapes of the data in each batch:

.. code-block:: python

    # Interpolation showcase using the data_loaders
    for batch_idx, batch_data in enumerate(data_loaders['train']):
        # batch_data is a dictionary with keys 'screen', 'responses'
        screen_data = batch_data['screen']
        responses = batch_data['responses']
        
        # Print or inspect the batch
        print(f"Batch {batch_idx}:")
        print("Screen Data:", screen_data.shape)
        print("Responses:", responses.shape)
        break

This will output something like:

.. code-block:: text

    Batch 0:
    Screen Data: torch.Size([15, 1, 32, 144, 144])
    Responses: torch.Size([15, 32, 12])

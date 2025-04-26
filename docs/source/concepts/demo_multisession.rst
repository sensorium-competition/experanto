Loading Multiple Sessions
=========================

To load multiple sessions at once, you can use the ``get_multisession_dataloader`` function from ``experanto.dataloaders``.

This function takes:

- A list of paths pointing to your experiment directories
- A configuration dictionary, similar to the one used for loading a single dataset

It returns a dictionary of ``MultiepochDataloader`` objects, each corresponding to a session, loaded with the specified configurations.

Example
-------

.. code-block:: python

    import os
    from experanto.dataloaders import get_multisession_dataloader

    from omegaconf import OmegaConf, open_dict
    from experanto.configs import DEFAULT_CONFIG as cfg

    cfg.dataset.modality_config.screen.transforms.Resize.size = [144,144] 
    cfg.dataset.modality_config.screen.interpolation.rescale_size = [144, 144]
    cfg.dataset.modality_config.screen.transforms.greyscale = True

    # Define paths to session folders
    parent_folder = '../data/allen_data'
    full_paths = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    # Load first two sessions
    train_dl = get_multisession_dataloader(full_paths[:2], cfg)

The returned ``train_dl`` is a dictionary containing two ``MultiepochDataloader`` objects. These can be further wrapped into standard ``torch.utils.data.DataLoader`` objects for use in training or evaluation workflows.

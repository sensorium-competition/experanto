Loading multiple sessions
=========================

To load multiple sessions at once, you can use :func:`~experanto.dataloaders.get_multisession_dataloader`.

This function takes:

- A list of paths pointing to your experiment directories
- A configuration dictionary, similar to the one used for loading a single dataset

It returns a :class:`~experanto.utils.LongCycler`, an iterator that wraps an
internal dictionary of :class:`~experanto.utils.MultiEpochsDataLoader` objects
(one per session) and yields ``(session_key, batch)`` pairs on each iteration,
cycling until the longest session is exhausted.

Example
-------

.. code-block:: python

    import os
    from experanto.dataloaders import get_multisession_dataloader
    from experanto.configs import DEFAULT_CONFIG as cfg

    cfg.dataset.modality_config.screen.transforms.Resize.size = [144, 144] 
    cfg.dataset.modality_config.screen.interpolation.rescale_size = [144, 144]
    cfg.dataset.modality_config.screen.transforms.greyscale = True

    # Define paths to session folders
    parent_folder = '../data/allen_data'
    full_paths = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    # Load first two sessions
    train_dl = get_multisession_dataloader(full_paths[:2], cfg)

    # Iterate — each step yields a (session_key, batch) tuple
    for session_key, batch in train_dl:
        print(session_key, batch['responses'].shape)

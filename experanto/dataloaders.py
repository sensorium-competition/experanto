from .data import ChunkDataset, DEFAULT_INTERP_CONFIG
from .utils import MultiEpochsDataLoader, LongCycler

def get_multisession_dataloaders(
        paths: list[str],
        interp_config: dict = DEFAULT_INTERP_CONFIG,
        chunk_size: int = 32,
        sampling_rate: int = 8,
        tier: str = "train",
        sample_stride: int = 4,
        include_blanks: bool = True,
        batch_size: int = 8,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool =True,
        prefetch_factor: int = 1,
):
    """
    Returns a dictionary of dataloaders for each session
    """
    dataloaders = {}
    for i, path in enumerate(paths):
        dataset = ChunkDataset(
            path,
            interp_config=interp_config,
            chunk_size=chunk_size,
            sampling_rate=sampling_rate,
            tier=tier,
            sample_stride=sample_stride,
            include_blanks=include_blanks,
        )
        dataloaders[i] = MultiEpochsDataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=drop_last,
                                                  prefetch_factor=prefetch_factor,
                                               )
    return LongCycler(dataloaders)
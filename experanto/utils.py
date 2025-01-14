import numpy as np
import torch
from torchvision.transforms import functional as F
import torch.nn.functional as pad_F

def linear_interpolate_1d_sequence(row, times_old, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [Neurons, times]
        times: old time points [Neurons, times] or [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced (inplace).
    """
    if keep_nans:
        interpolated_array = np.interp(times_new, times_old, row)
    else:
        # Find indices of non-NaN values
        valid_indices = np.where(~np.isnan(row))[0]
        valid_times = times_old[valid_indices]
        # Interpolate the column using linear interpolation
        interpolated_array = np.interp(times_new, valid_times, row[valid_indices])
    return interpolated_array


def linear_interpolate_sequences(array, times, times_new, keep_nans=False):
    """
    Interpolates columns in a NumPy array and replaces NaNs with interpolated values

    Args:
        array: The input NumPy array [times, ch]
        times: old time points  [times]
        times_new:  new time points [times2]
        keep_nans:  if we want to keep and return nans after interpolation

    Returns:
        The interpolated array with NaNs replaced.
    """
    array = array.T
    if array.shape[0] == 1:
        return linear_interpolate_1d_sequence(
            array.T.flatten(), times, times_new, keep_nans=keep_nans
        )
    interpolated_array = np.full((array.shape[0], len(times_new)), np.nan)
    for row_idx, row in enumerate(array):
        interpolated_array[row_idx] = linear_interpolate_1d_sequence(
            row, times, times_new, keep_nans=keep_nans
        )
    return interpolated_array.T


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ solves bug to keep all workers initialized across epochs.
    From https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778
    and
    https://github.com/huggingface/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
    """
    def __init__(self, *args, shuffle_each_epoch=True, **kwargs, ):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        self.shuffle_each_epoch = shuffle_each_epoch

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        if self.shuffle_each_epoch and hasattr(self.dataset, "shuffle_valid_screen_times"):
            self.dataset.shuffle_valid_screen_times()
        for i in range(len(self)):
            yield next(self.iterator)




class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# borrowed with <3 from
# https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/training/cyclers.py
def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Exhauster:
    """
    Given a dictionary of data loaders, mapping data_key into a data loader, steps through each data loader, moving onto the next data loader
    only upon exhausing the content of the current data loader.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for data_key, loader in self.loaders.items():
            for batch in loader:
                yield data_key, batch

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.min_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.min_batches

class GazeBasedCrop(torch.nn.Module):
    def __init__(self, crop_size, pixel_per_degree):
        """
        Custom transform to crop an image based on a given center point in degrees
        and pad it to maintain the target crop size.

        Args:
            crop_size (tuple): (height, width) of the crop in pixels.
            pixel_per_degree (float): Conversion factor from degrees to pixels.
        """
        super().__init__()
        self.crop_size = crop_size
        self.pixel_per_degree = pixel_per_degree

    def forward(self, inputs):
        """
        Perform cropping and padding based on the gaze center and ensure consistent output size.

        Args:
            inputs (tuple): A tuple of (image, center), where:
                - image (Tensor): Input image tensor of shape (C, H, W).
                - center (tuple): (x, y) coordinates of the gaze point in degrees.

        Returns:
            Tensor: Cropped and padded image tensor of shape (C, target_height, target_width).
        """
        image, center = inputs
        h, w = self.crop_size

        # Ensure center is unpacked into Python scalars
        if isinstance(center, torch.Tensor) and center.numel() == 2:
            x_deg, y_deg = center.tolist()
        elif isinstance(center, (list, tuple)):
            x_deg, y_deg = center
        else:
            raise ValueError(f"Unexpected center format: {type(center)}, {center}")

        # Convert gaze points from degrees to pixels
        x_px = x_deg * self.pixel_per_degree
        y_px = y_deg * self.pixel_per_degree

        # Compute crop bounds
        left = max(0, int(x_px - w // 2))
        top = max(0, int(y_px - h // 2))
        right = min(image.shape[-1], left + w)  # Ensure crop stays within bounds
        bottom = min(image.shape[-2], top + h)

        # Crop the image
        cropped_image = F.crop(image, top, left, bottom - top, right - left)

        # Pad the cropped image to ensure consistent size
        cropped_image = self._pad_to_size(cropped_image, (h, w))

        return cropped_image

    def _pad_to_size(self, image, size):
        """
        Pad an image to the desired size, ensuring consistent dimensions.

        Args:
            image (Tensor): Input image tensor of shape (C, H, W) or (1, C, H, W).
            size (tuple): Desired output size (height, width).

        Returns:
            Tensor: Padded image of shape (C, target_height, target_width).
        """
        # Remove any batch dimension if present
        if image.ndim == 4 and image.size(0) == 1:
            image = image.squeeze(0)

        if image.ndim != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected (C, H, W).")

        _, h, w = image.shape
        target_h, target_w = size

        # Calculate required padding for each side
        pad_h = target_h - h
        pad_w = target_w - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Ensure padding is non-negative
        pad_top = max(0, pad_top)
        pad_bottom = max(0, pad_bottom)
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)

        # Apply padding
        return pad_F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

def replace_nans_with_neighbors(data):
    """
    Replace NaN values in a tensor with the previous value if available,
    otherwise with the next valid value.

    Args:
        data (torch.Tensor): 2D tensor of shape (N, 2) where NaNs may exist.

    Returns:
        torch.Tensor: Tensor with NaNs replaced.
    """
    data = data.clone()  # Create a copy to avoid modifying the original tensor
    for i in range(data.size(0)):  # Iterate over rows
        if torch.isnan(data[i]).any():
            if i > 0 and not torch.isnan(data[i - 1]).any():  # Use previous value
                data[i] = data[i - 1]
            elif i < data.size(0) - 1 and not torch.isnan(data[i + 1]).any():  # Use next value
                data[i] = data[i + 1]
            else:
                # If no previous or next valid value, replace with zeros or a default value
                data[i] = torch.zeros_like(data[i])
    return data
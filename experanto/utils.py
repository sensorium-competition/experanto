import numpy as np
import torch
from torchvision.transforms import functional as F
import torch.nn.functional as pad_F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def get_validation_split(n_images, train_frac, seed):
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n_images: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    if seed:
        np.random.seed(seed)
    train_idx, val_idx = np.split(
        np.random.permutation(int(n_images)), [int(n_images * train_frac)]
    )
    assert not np.any(
        np.isin(train_idx, val_idx)
    ), "train_set and val_set are overlapping sets"

    return train_idx, val_idx

class GazeBasedCrop(torch.nn.Module):
    def __init__(self, crop_size, pixel_per_degree, monitor_center, dest_rect, bg_color=(127.5, 127.5, 127.5)):
        """
        Crops an image centered on the receptive field (RF) location, adjusted for gaze.
        The image is first resized to `dest_rect` and then padded to match the full monitor size.

        Args:
            crop_size (tuple): (height, width) of the crop in pixels.
            pixel_per_degree (float): Conversion factor from degrees to pixels.
            monitor_center (tuple): (x, y) pixel coordinates of the monitor center.
            dest_rect (tuple): (x1, y1, x2, y2) defining stimulus area (where stimulus was shown).
            bg_color (tuple): Background color in RGB (default is gray: 127.5, 127.5, 127.5).
        """
        super().__init__()
        self.crop_size = crop_size
        self.pixel_per_degree = pixel_per_degree
        self.monitor_size = (monitor_center[0] * 2, monitor_center[1] * 2)  # Compute monitor size
        self.dest_rect = dest_rect  # Resize image to match this region before padding
        self.bg_color = bg_color  # Background color (default gray)

    def forward(self, inputs, gaze, fix_spot, stim_location, dynamic=False):
        """
        Perform resizing, padding, and cropping based on gaze-adjusted RF location.

        Args:
            inputs (Tensor): Input image tensor of shape (C, H, W).
            gaze (tuple or tensor): (x, y) current gaze position in degrees.
            fix_spot (tuple): (x, y) fixation spot location in degrees.
            stim_location (tuple): (x, y) RF center location in degrees.
            dynamic (bool): If True, updates RF center per frame. If False, uses static mean gaze.

        Returns:
            Tensor: Cropped and padded image of shape (C, target_height, target_width).
        """
        image = self._ensure_correct_channels(inputs)
        h, w = self.crop_size

        # Step 1: Resize image to match `destRect`
        dest_width = self.dest_rect[2] - self.dest_rect[0]
        dest_height = self.dest_rect[3] - self.dest_rect[1]
        image = F.resize(image, (dest_height, dest_width))  # Resize to `destRect`
        #plt.imshow(image[0])
        # Step 2: Pad image to match the full monitor size
        image = self._pad_to_monitor_size(image)

        # Step 3: Compute RF center offset based on gaze and fixation spot
        rf_center_x, rf_center_y = self._compute_rf_center(gaze, fix_spot, stim_location, dynamic)

        # Convert RF center from degrees to pixels (Monitor center is now origin)
        x_px = rf_center_x  + self.monitor_size[0] // 2
        y_px = rf_center_y  + self.monitor_size[1] // 2  # Flip y-axis

        # Compute crop bounds
        left = int(x_px - w // 2)
        top = int(y_px - h // 2)
        right = left + w
        bottom = top + h

        # Step 4: Crop and pad if necessary

        #self._plot_debug(image, x_px, y_px, left, top, right, bottom, fix_spot, stim_location, gaze)
        cropped_image = self._safe_crop(image, left, top, right, bottom)
        return cropped_image

    def _compute_rf_center(self, gaze, fix_spot, stim_location, dynamic):
        """
        Computes the receptive field center adjusted by gaze offset.

        Args:
            gaze (tuple): (x, y) gaze position in degrees.
            fix_spot (tuple): (x, y) fixation spot in degrees.
            stim_location (tuple): (x, y) RF center location in degrees.
            dynamic (bool): Whether to use frame-by-frame gaze shift or static mean gaze.

        Returns:
            tuple: (rf_center_x, rf_center_y) adjusted RF location in degrees.
        """
        gaze = gaze * self.pixel_per_degree
        if dynamic:
            rf_center_x = stim_location[0] + (gaze[0] - fix_spot[0])
            rf_center_y = stim_location[1] + (gaze[1] - fix_spot[1])
        else:
            mean_gaze_x, mean_gaze_y = torch.mean(torch.tensor(gaze), dim=0)
            rf_center_x = stim_location[0] + (mean_gaze_x - fix_spot[0])
            rf_center_y = stim_location[1] + (mean_gaze_y - fix_spot[1])

        return rf_center_x, rf_center_y

    def _pad_to_monitor_size(self, image):
        """
        Pads the resized image to match the full monitor size using `bg_color`.

        Args:
            image (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Padded image of shape (C, monitor_height, monitor_width).
        """
        C, H, W = image.shape
        monitor_w, monitor_h = self.monitor_size

        # Compute padding
        pad_left = (monitor_w - W) // 2
        pad_right = monitor_w - W - pad_left
        pad_top = (monitor_h - H) // 2
        pad_bottom = monitor_h - H - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return self._apply_padding(image, padding, C)

    def _safe_crop(self, image, left, top, right, bottom):
        """
        Crops and pads the image with a background color if necessary.

        Args:
            image (Tensor): Input image tensor (C, H, W).
            left, top, right, bottom: Cropping bounds.

        Returns:
            Tensor: Cropped and padded image of size (C, target_height, target_width).
        """
        C, H, W = image.shape  # Get image dimensions
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(W, right)
        crop_bottom = min(H, bottom)

        # Crop the valid part of the image
        cropped = F.crop(image, crop_top, crop_left, crop_bottom - crop_top, crop_right - crop_left)
        #plt.imshow(cropped[0])

        # Compute padding amounts
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - W)
        pad_bottom = max(0, bottom - H)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            cropped = self._apply_padding(cropped, padding, C)

        return cropped

    def _apply_padding(self, image, padding, C):
        """
        Apply padding to the image while maintaining correct data types.
        
        Args:
            image (Tensor): Image tensor of shape (C, H, W).
            padding (tuple): Padding values (left, right, top, bottom).
            C (int): Number of channels (1 for grayscale, 3 for RGB).
        
        Returns:
            Tensor: Padded image.
        """
        # Convert padding values to integers
        padding = tuple(int(p) for p in padding)

        if C == 1:  # Grayscale image
            fill_value = float(self.bg_color[0])
        else:  # RGB image
            fill_value = tuple(int(c) for c in self.bg_color)  # Convert RGB values to int
        
        return F.pad(image, padding, fill=fill_value)

    def _ensure_correct_channels(self, image):
        """
        Ensures the image has the correct shape (C, H, W).

        Args:
            image (Tensor): Input image tensor.

        Returns:
            Tensor: Image in (C, H, W) format.
        """
        if image.ndim == 2:  # If grayscale (H, W), add a channel
            image = image.unsqueeze(0)  # Convert to (1, H, W)
        elif image.ndim == 3 and image.shape[-1] == 3:  # If (H, W, 3), permute to (3, H, W)
            image = image.permute(2, 0, 1)
        return image

    def _plot_debug(self, image, x_px, y_px, left, top, right, bottom, fix_spot, stim_location, gaze):
        """
        Debugging function to visualize the cropping process.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap="gray", origin="upper")  # Convert CHW -> HWC

        # Plot important points
        ax.scatter(x_px, y_px, color = 'blue',marker="+", s=80, label="RF Center (x_px, y_px)")
        print(x_px,y_px, stim_location)
        ax.scatter(stim_location[0]  + self.monitor_size[0] // 2,
                   stim_location[1]  + self.monitor_size[1] // 2, 
                   color="red", marker="+", s=80, label="Stimulus Location")
        ax.scatter(np.array(gaze[:,0]*self.pixel_per_degree) + self.monitor_size[0] // 2,
                   np.array(gaze[:,1]*self.pixel_per_degree) + self.monitor_size[1] // 2,
                   color="blue", marker="*", s=20, label="Gaze Location")
        print(np.array(gaze[:,0])* self.pixel_per_degree  + self.monitor_size[0] // 2,
                   np.array(gaze[:,1])* self.pixel_per_degree  + self.monitor_size[1] // 2)
        ax.scatter(fix_spot[0] + self.monitor_size[0] // 2,
                   fix_spot[1]  + self.monitor_size[1] // 2, 
                   color="red", marker="*", s=80, label="Fixation Spot")
        ax.scatter(self.monitor_size[0] // 2, self.monitor_size[1] // 2, 
                   color="green", marker="o", s=40, label="Monitor Center")

        # Plot bounding box
        rect = patches.Rectangle(
            (left, top), right - left, bottom - top,
            linewidth=1, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

        ax.legend()
        plt.title("Debug Visualization: RF Center, Fixation Spot, and Crop Area")
        plt.show()
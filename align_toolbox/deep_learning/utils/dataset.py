import datetime
import os

import cv2
import numpy as np
import pandas as pd
import torch
from cv2 import resize
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from align_toolbox.data_analysis.time_series import (
    crop_series_to_length,
    pad_series_to_length,
)
from align_toolbox.deep_learning.utils.augmentation import (
    get_prediction_augmentation,
    get_training_augmentation,
)
from align_toolbox.deep_learning.utils.util import (
    get_closest_lower_multiple,
    get_closest_upper_multiple,
)
from align_toolbox.foundation import image_handling


class TiledSegmentationDataset(Dataset):
    """
    PyTorch Dataset for tiled segmentation training.

    Loads image and mask pairs from file paths stored in a DataFrame and returns a
    randomly selected tile from each sample. Tile boundaries are pre-computed using
    ``pytorch_toolbelt.inference.ImageSlicer``.

    Parameters:
        dataset (pd.DataFrame): DataFrame with at least ``image_column`` and
            ``mask_column`` columns containing file paths.
        image_slicers (dict): Mapping from image shape to ``ImageSlicer`` instance.
        channels (int or list[int]): Channel indices to load from the image.
        mask_column (str, optional): DataFrame column name for mask paths.
            (default: ``"mask"``)
        image_column (str, optional): DataFrame column name for image paths.
            (default: ``"image"``)
        transform (callable, optional): MONAI Compose transform applied to the
            ``{"image": ..., "mask": ...}`` dictionary. (default: None)
    """

    def __init__(
        self,
        dataset,
        image_slicers,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        if not isinstance(channels, list):
            channels = [channels]
        self.channels = channels
        self.image_slicers = image_slicers
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], self.channels)
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform({"image": img, "mask": mask})
            img = transformed["image"]
            mask = transformed["mask"]

        # switch the axes to go from (C, H, W) to (H, W, C) if necessary
        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = img[..., np.newaxis]

        slicer = self.image_slicers[img.shape]

        tiles = slicer.split(img)

        tiles_ground_truth = slicer.split(mask)

        selected_tile = np.random.randint(0, len(tiles))
        img = tiles[selected_tile]
        mask = tiles_ground_truth[selected_tile]
        mask = mask[np.newaxis, ...]

        img = np.transpose(img, (2, 0, 1))  # switch back to (C, H, W)

        return img.astype(np.float32), mask


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for full-image segmentation training.

    Loads image and mask pairs from file paths stored in a DataFrame. Images are
    resized (padded or cropped) in the collate function to a divisibility-enforced
    common size within each batch.

    Parameters:
        dataset (pd.DataFrame): DataFrame with ``image_column`` and ``mask_column``
            columns containing file paths.
        channels (int or list[int]): Channel indices to load from the image.
        mask_column (str, optional): Column name for mask paths. (default: ``"mask"``)
        image_column (str, optional): Column name for image paths. (default: ``"image"``)
        transform (callable, optional): MONAI Compose transform. (default: None)
        enforce_divisibility_by (int, optional): Batch images are resized so their
            spatial dimensions are multiples of this value. (default: 32)
        pad_or_crop (str, optional): Whether to pad (``"pad"``) or crop (``"crop"``)
            images to the common batch size. (default: ``"pad"``)
        mask_pad_value (int, optional): Fill value used when padding masks.
            (default: -1)
    """

    def __init__(
        self,
        dataset,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
        enforce_divisibility_by=32,
        pad_or_crop="pad",
        mask_pad_value=-1,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        if not isinstance(channels, list):
            channels = [channels]
        self.channels = channels
        self.transform = transform
        if enforce_divisibility_by is None:
            enforce_divisibility_by = 1
        self.enforce_divisibility_by = enforce_divisibility_by
        if pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be either 'pad' or 'crop'")
        self.pad_or_crop = pad_or_crop

        if pad_or_crop == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.mask_resize_function = (
                lambda dim, new_dim_x, new_dim_y: image_handling.pad_to_dim_equally(
                    dim, new_dim_x, new_dim_y, pad_value=mask_pad_value
                )
            )
            self.multiplier_function = get_closest_upper_multiple
        else:
            self.resize_function = image_handling.crop_to_dim_equally
            self.mask_resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], self.channels)
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform({"image": img, "mask": mask})
            img = transformed["image"]
            mask = transformed["mask"]

        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return img.astype(np.float32), mask

    def collate_fn(self, batch):
        imgs, masks = zip(*batch)

        original_shapes = [img.shape for img in imgs]

        # find the maximum dimensions in the batch
        if self.pad_or_crop == "pad":
            dim_x = max([shape[-2] for shape in original_shapes])
            dim_y = max([shape[-1] for shape in original_shapes])
        else:
            dim_x = min([shape[-2] for shape in original_shapes])
            dim_y = min([shape[-1] for shape in original_shapes])

        # find the new dimensions
        new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
        new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

        # resize the images
        resized_images = []
        resized_masks = []
        for img, mask in zip(imgs, masks):
            resized_images.append(self.resize_function(img, new_dim_x, new_dim_y))
            resized_masks.append(self.mask_resize_function(mask, new_dim_x, new_dim_y))

        resized_images = torch.tensor(np.array(resized_images), dtype=torch.float32)
        resized_masks = torch.tensor(np.array(resized_masks), dtype=torch.float32)

        return resized_images, resized_masks


class SegmentationPredictionDataset(Dataset):
    """
    PyTorch Dataset for segmentation inference.

    Loads images from a list of file paths, optionally rescales them, and pads or
    crops batches to a common size that is a multiple of ``enforce_divisibility_by``.
    The collate function returns image paths, resized tensors, original shapes, and
    indices of images that failed to load.

    Parameters:
        image_paths (list[str]): Paths to the image files.
        channels (int or list[int]): Channel indices to load.
        transform (callable, optional): MONAI Compose transform. (default: None)
        enforce_divisibility_by (int, optional): Images are resized so their spatial
            dimensions are multiples of this value. (default: 32)
        scale_factor (float, optional): Isotropic rescaling factor applied to each
            image before batching. (default: 1.0)
        pad_or_crop (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
    """

    def __init__(
        self,
        image_paths,
        channels,
        transform=None,
        enforce_divisibility_by=32,
        scale_factor=1.0,
        pad_or_crop="pad",
    ):
        self.images = image_paths
        if not isinstance(channels, list):
            channels = [channels]
        self.channels = channels
        self.transform = transform
        if enforce_divisibility_by is None:
            enforce_divisibility_by = 1
        self.enforce_divisibility_by = enforce_divisibility_by
        if pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be either 'pad' or 'crop'")

        self.pad_or_crop = pad_or_crop

        if pad_or_crop == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.multiplier_function = get_closest_upper_multiple
        else:
            self.resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]

        try:
            img = image_handling.read_tiff_file(img_path, self.channels)

            if self.transform is not None:
                transformed = self.transform({"image": img})
                img = transformed["image"]

            if len(img.shape) == 2:
                img = img[np.newaxis, ...]

            if self.scale_factor != 1.0:
                scale = tuple([self.scale_factor] * 2 + [1] * (img.ndim - 2))
                img = np.moveaxis(img, [-2, -1], [0, 1])
                img = rescale(
                    img,
                    scale=scale,
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.moveaxis(img, [0, 1], [-2, -1])

            return img_path, img.astype(np.float32), img.shape
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return img_path, None, None

    def collate_fn(self, batch):
        img_paths, imgs, original_shapes = zip(*batch)

        invalid_indices = [j for j, img in enumerate(imgs) if img is None]
        if len(invalid_indices) == len(imgs):
            return None, None, None, None

        img_paths, imgs, original_shapes = (
            [img_paths[j] for j in range(len(imgs)) if j not in invalid_indices],
            [imgs[j] for j in range(len(imgs)) if j not in invalid_indices],
            [original_shapes[j] for j in range(len(imgs)) if j not in invalid_indices],
        )

        # find the maximum dimensions in the batch
        if self.pad_or_crop == "pad":
            dim_x = max([shape[-2] for shape in original_shapes])
            dim_y = max([shape[-1] for shape in original_shapes])
        else:
            dim_x = min([shape[-2] for shape in original_shapes])
            dim_y = min([shape[-1] for shape in original_shapes])

        # find the new dimensions
        new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
        new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

        # resize the images
        resized_images = []
        for img in imgs:
            resized_images.append(self.resize_function(img, new_dim_x, new_dim_y))
        resized_images = torch.tensor(np.array(resized_images), dtype=torch.float32)
        return img_paths, resized_images, original_shapes, invalid_indices


class StackPredictionDataset(Dataset):
    """
    PyTorch Dataset for plane-by-plane inference on a z-stack.

    Accepts a z-stack as a NumPy array or a file path. Optionally downscales
    planes before loading. Each ``__getitem__`` call returns a single plane,
    ready for inference.

    Parameters:
        stack (str or np.ndarray): z-stack as an array of shape ``(N, H, W)`` or
            ``(N, C, H, W)``, or a path to a TIFF file.
        channels (int, list[int], or None): Channel indices to load when ``stack``
            is a file path.
        transform (callable, optional): MONAI Compose transform applied per plane.
            (default: None)
        enforce_divisibility_by (int, optional): Spatial dimensions are resized to
            multiples of this value. (default: 32)
        pad_or_crop (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
        scale_factor (float, optional): Isotropic rescaling factor applied to each
            plane via ``cv2.resize``. (default: 1.0)
    """

    def __init__(
        self,
        stack,
        channels,
        transform=None,
        enforce_divisibility_by=32,
        pad_or_crop="pad",
        scale_factor=1.0,
    ):
        if not isinstance(channels, list) and channels is not None:
            channels = [channels]
        if isinstance(stack, str):
            stack = image_handling.read_tiff_file(stack, channels_to_keep=channels)

        self.channels = channels
        self.transform = transform
        self.enforce_divisibility_by = enforce_divisibility_by
        if pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be either 'pad' or 'crop'")
        self.pad_or_crop = pad_or_crop
        if pad_or_crop == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.multiplier_function = get_closest_upper_multiple
        else:
            self.resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple
        self.scale_factor = scale_factor

        if self.scale_factor != 1.0:
            binned_stack = []
            if stack.ndim == 3:
                for plane in stack:
                    binned_plane = resize(
                        plane,
                        (
                            int(plane.shape[1] * self.scale_factor),
                            int(plane.shape[0] * self.scale_factor),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    binned_stack.append(binned_plane)
            elif stack.ndim == 4:
                for channel in stack:
                    binned_channel = []
                    for plane in channel:
                        binned_plane = resize(
                            plane,
                            (
                                int(plane.shape[1] * self.scale_factor),
                                int(plane.shape[0] * self.scale_factor),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
                        binned_channel.append(binned_plane)
                    binned_channel = np.stack(binned_channel, axis=0)
                    binned_stack.append(binned_channel)

            stack = np.stack(binned_stack, axis=0)

        self.stack_shape = stack.shape
        new_x_dim = self.multiplier_function(
            self.stack_shape[-2], self.enforce_divisibility_by
        )
        new_y_dim = self.multiplier_function(
            self.stack_shape[-1], self.enforce_divisibility_by
        )
        self.stack = self.resize_function(stack, new_x_dim, new_y_dim)

    def __len__(self):
        return self.stack_shape[0]

    def __getitem__(self, i):
        plane = self.stack[i]
        if self.transform is not None:
            transformed = self.transform({"image": plane})
            plane = transformed["image"]
        if len(plane.shape) == 2:
            plane = plane[np.newaxis, ...]
        return plane.astype(np.float32)


class ClassificationDataset(Dataset):
    """
    PyTorch Dataset for image classification training.

    Loads images from file paths and their corresponding class labels. For
    multi-class problems (n_classes > 2) labels are one-hot encoded.

    Parameters:
        dataset (pd.DataFrame): DataFrame with ``image_column`` and
            ``class_column`` columns.
        channels (int or list[int]): Channel index (or indices) to load.
        n_classes (int): Total number of classes. Labels are one-hot encoded
            when n_classes > 2.
        class_column (str, optional): Column name for class labels.
            (default: ``"class"``)
        image_column (str, optional): Column name for image paths.
            (default: ``"image"``)
        transform (callable, optional): MONAI Compose transform applied to
            ``{"image": ...}``. (default: None)
    """

    def __init__(
        self,
        dataset,
        channels,
        n_classes,
        class_column="class",
        image_column="image",
        transform=None,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[class_column].values.astype(float).tolist()
        self.channels = channels
        self.transform = transform

        # convert ground truth to one-hot encoding
        if n_classes > 2:
            labels = np.asarray(self.ground_truth).astype(int)
            self.ground_truth = np.eye(n_classes)[labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        class_value = self.ground_truth[i]

        if self.transform is not None:
            transformed = self.transform({"image": img})
            img = transformed["image"]

        if len(img.shape) == 2:
            img = img[np.newaxis, ...]

        return img.astype(np.float32), class_value


class QualityControlDataset(Dataset):
    """
    PyTorch Dataset for quality-control classification training.

    Loads image + mask pairs and their quality labels. When ``mask_paths`` is
    provided, image and mask are concatenated along the channel axis before the
    transform. The collate function discards samples whose mask has no foreground.

    Parameters:
        image_paths (list[str]): Paths to image files.
        mask_paths (list[str] or None): Paths to mask files. Pass ``None`` or an
            empty list for image-only mode.
        channels (int or list[int]): Channel indices to load from images.
        labels (list): Class labels (integers or strings matching ``classes``).
        classes (list): Ordered list of class names.
        enforce_divisibility_by (int, optional): Batch spatial dimensions are
            resized to multiples of this value. (default: 32)
        resize_method (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
        transform (callable, optional): MONAI Compose transform. (default: None)
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        channels,
        labels,
        classes,
        enforce_divisibility_by=32,
        resize_method="pad",
        transform=None,
    ):
        self.images = image_paths
        self.image_only = (mask_paths is None) or (len(mask_paths) == 0)
        self.masks = mask_paths
        if not isinstance(channels, list):
            channels = [channels]
        self.channels = channels

        classes = list(classes)
        if isinstance(labels[0], str):
            label_mapping = {cls: i for i, cls in enumerate(classes)}
            labels = [label_mapping[label] for label in labels]

        self.labels = labels

        self.classes = classes
        if enforce_divisibility_by is None:
            enforce_divisibility_by = 1
        self.enforce_divisibility_by = enforce_divisibility_by

        self.resize_method = resize_method
        if resize_method == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.multiplier_function = get_closest_upper_multiple
        elif resize_method == "crop":
            self.resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

        self.transform = transform
        self.n_classes = len(classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], self.channels)

        if self.image_only:
            if self.transform is not None:
                transformed = self.transform({"image": img})
                img = transformed["image"]
            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
            return img, self.labels[i]

        else:
            mask = image_handling.read_tiff_file(self.masks[i])

            if img.shape != mask.shape:
                # pad the smaller one to match the larger one
                img, mask = image_handling.pad_images_to_same_dim(img, mask)
            label = self.labels[i]
            if self.transform is not None:
                transformed = self.transform({"image": img, "mask": mask})
                img = transformed["image"]
                mask = transformed["mask"]

            if len(img.shape) == 2:
                img = img[np.newaxis, ...]
            if len(mask.shape) == 2:
                mask = mask[np.newaxis, ...]
            combined_imgs = np.concatenate([img, mask], axis=0)
            # combined_imgs = img.copy()
            return combined_imgs, label

    def collate_fn(self, batch):
        # for training, we can simply remove any masks that have no foreground
        combined_imgs, labels = zip(*batch)

        valid_indices = [
            i for i, img in enumerate(combined_imgs) if np.sum(img[-1]) > 0
        ]

        if len(valid_indices) <= 1:
            return None
        else:
            combined_imgs = [combined_imgs[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

        original_shapes = [img.shape for img in combined_imgs]

        # find the maximum dimensions in the batch
        # images will have wildly different sizes, so we cap the maximum size to avoid OOM
        # we also enforce a minimum size to avoid too much cropping
        MAX_DIM_X = 2048
        MIN_DIM_X = 32
        MAX_DIM_Y = 2048
        MIN_DIM_Y = 64

        if self.resize_method == "pad":
            dim_x = max([shape[-2] for shape in original_shapes])
            dim_y = max([shape[-1] for shape in original_shapes])

            dim_x = min(dim_x, MAX_DIM_X)
            dim_y = min(dim_y, MAX_DIM_Y)
        else:
            dim_x = min([shape[-2] for shape in original_shapes])
            dim_y = min([shape[-1] for shape in original_shapes])

            dim_x = max(dim_x, MIN_DIM_X)
            dim_y = max(dim_y, MIN_DIM_Y)

        # find the new dimensions
        new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
        new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

        # resize the images
        resized_images = []
        for img in combined_imgs:
            img_h, img_w = img.shape[-2], img.shape[-1]

            if img_h > new_dim_x or img_w > new_dim_y:
                img = image_handling.crop_to_dim_equally(
                    img, min(img_h, new_dim_x), min(img_w, new_dim_y)
                )

            if img_h < new_dim_x or img_w < new_dim_y:
                img = image_handling.pad_to_dim_equally(
                    img, new_dim_x, new_dim_y, pad_value=0
                )

            resized_images.append(img)

        resized_images = torch.tensor(np.array(resized_images), dtype=torch.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.long)

        return resized_images, labels


class QualityControlPredictionDataset(Dataset):
    """
    PyTorch Dataset for quality-control classification inference.

    Loads image + mask pairs and concatenates them along the channel axis.
    The collate function tracks samples rejected due to empty or failed masks
    and returns their indices alongside valid batches.

    Parameters:
        image_paths (list[str]): Paths to image files.
        mask_paths (list[str]): Paths to mask files.
        channels (int or list[int]): Channel indices to load from images.
        enforce_divisibility_by (int, optional): Batch spatial dimensions are
            resized to multiples of this value. (default: 32)
        resize_method (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
        transform (callable, optional): MONAI Compose transform. (default: None)
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        channels,
        enforce_divisibility_by=32,
        resize_method="pad",
        transform=None,
    ):
        self.images = image_paths
        self.masks = mask_paths
        if not isinstance(channels, list):
            channels = [channels]
        self.channels = channels
        if enforce_divisibility_by is None:
            enforce_divisibility_by = 1
        self.enforce_divisibility_by = enforce_divisibility_by

        self.resize_method = resize_method
        if resize_method == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.multiplier_function = get_closest_upper_multiple
        elif resize_method == "crop":
            self.resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        try:
            img = image_handling.read_tiff_file(self.images[i], self.channels)
            mask = image_handling.read_tiff_file(self.masks[i])

            if img.shape != mask.shape:
                # pad the smaller one to match the larger one
                img, mask = image_handling.pad_images_to_same_dim(img, mask)

            if self.transform is not None:
                transformed = self.transform({"image": img, "mask": mask})
                img = transformed["image"]
                mask = transformed["mask"]

            if len(img.shape) == 2:
                img = img[np.newaxis, ...]

            if len(mask.shape) == 2:
                mask = mask[np.newaxis, ...]
            combined_imgs = np.concatenate([img, mask], axis=0)

            return combined_imgs.astype(np.float32)
        except Exception as e:
            print(f"Error loading image or mask {self.images[i]}, {self.masks[i]}: {e}")
            return None

    def collate_fn(self, batch):
        # for prediction, we need to keep track of which masks have no foreground, to automatically mark them as unusable
        combined_imgs = batch

        valid_indices = [
            i
            for i, img in enumerate(combined_imgs)
            if (img is not None and np.sum(img[-1]) > 0)
        ]
        rejected_indices = [
            i for i in range(len(combined_imgs)) if i not in valid_indices
        ]
        combined_imgs = [combined_imgs[i] for i in valid_indices]

        original_shapes = [img.shape for img in combined_imgs]

        # find the maximum dimensions in the batch
        if self.resize_method == "pad":
            dim_x = max([shape[-2] for shape in original_shapes])
            dim_y = max([shape[-1] for shape in original_shapes])
        else:
            dim_x = min([shape[-2] for shape in original_shapes])
            dim_y = min([shape[-1] for shape in original_shapes])

        # find the new dimensions
        new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
        new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

        # resize the images
        resized_images = []
        for img in combined_imgs:
            resized_img = self.resize_function(img, new_dim_x, new_dim_y)
            resized_images.append(resized_img)

        resized_images = torch.tensor(np.array(resized_images), dtype=torch.float32)

        return resized_images, rejected_indices


class KeypointDetection1DTrainingDataset(Dataset):
    """
    PyTorch Dataset for 1D keypoint detection training.

    Stores input time-series together with their target heatmaps and target
    keypoint indices. A per-keypoint presence target is derived on the fly from
    ``index_targets``: a keypoint counts as present when its index is not NaN.
    The collate function pads or crops all series in a batch to the same length
    (a multiple of ``enforce_divisibility_by``), builds a boolean mask marking
    the unpadded positions, and drops samples whose series contains NaN values.

    Parameters:
        inputs (array-like): Sequence of input series of shape ``(T,)`` or
            ``(C, T)``. 1D series are reshaped to ``(1, T)``.
        heatmap_targets (array-like): Sequence of target heatmaps of shape
            ``(T,)`` or ``(n_classes, T)``, aligned with ``inputs``.
        index_targets (array-like): Sequence of target keypoint indices of shape
            ``(n_classes,)`` or ``(1, n_classes)``, aligned with ``inputs``. NaN
            marks a keypoint that is absent from the series.
        enforce_divisibility_by (int, optional): Target batch length is rounded
            to a multiple of this value. Pass ``None`` to skip resizing entirely.
            (default: 64)
        resize_method (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
    """

    def __init__(
        self,
        inputs,
        heatmap_targets,
        index_targets,
        enforce_divisibility_by=64,
        resize_method="pad",
    ):
        self.input_series = inputs
        self.heatmap_targets = heatmap_targets
        self.index_targets = index_targets
        self.enforce_divisibility_by = enforce_divisibility_by

        self.resize_method = resize_method
        if resize_method == "pad":
            self.resize_function = pad_series_to_length
            self.target_resize_function = pad_series_to_length
            self.multiplier_function = get_closest_upper_multiple
        elif resize_method == "crop":
            self.resize_function = crop_series_to_length
            self.target_resize_function = crop_series_to_length
            self.multiplier_function = get_closest_lower_multiple

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, i):
        series = self.input_series[i]
        heatmap_target = self.heatmap_targets[i]
        index_target = self.index_targets[i]

        presence_target = (~np.isnan(np.array(index_target))).astype(np.float32)

        if series.ndim == 1:
            series = series.reshape(1, -1)
        if heatmap_target.ndim == 1:
            heatmap_target = heatmap_target.reshape(1, -1)
        if index_target.ndim == 1:
            index_target = index_target.reshape(1, -1)

        return (
            series.astype(np.float32),
            heatmap_target.astype(np.float32),
            index_target.astype(np.float32),
            presence_target,
            series.shape,
        )

    def collate_fn(self, batch):
        """
        Resize a batch of series to a common length and drop invalid samples.

        Parameters:
            batch (list[tuple]): Samples returned by ``__getitem__``.

        Returns:
            tuple: ``(series, valid_masks, heatmap_targets, index_targets,
            presence_targets)``, where ``series`` has shape ``(B, C, new_length)``,
            ``valid_masks`` has shape ``(B, new_length)`` and is ``True`` on the
            unpadded positions, and the targets are aligned with the kept samples.
            When ``enforce_divisibility_by`` is ``None``, the untouched
            ``(series, heatmap_targets, index_targets, original_shapes)`` tuple is
            returned instead.
        """
        series, heatmap_targets, index_targets, presence_targets, original_shapes = zip(
            *batch
        )

        if self.enforce_divisibility_by is None:
            return series, heatmap_targets, index_targets, original_shapes

        if self.resize_method == "crop":
            target_length = min([shape[-1] for shape in original_shapes])
        else:
            target_length = max([shape[-1] for shape in original_shapes])

        new_length = self.multiplier_function(
            target_length, self.enforce_divisibility_by
        )

        # resize the images
        resized_series = []
        valid_masks = []
        for series_i, original_length in zip(
            series, [shape[-1] for shape in original_shapes]
        ):
            resized_series_i = self.resize_function(series_i, new_length)

            # get the mask of unpadded values (i.e., the original values before padding)
            mask_i = np.zeros(new_length, dtype=bool)
            mask_i[: min(original_length, new_length)] = True
            valid_masks.append(mask_i)
            resized_series.append(resized_series_i)

        resized_series = np.array(resized_series, dtype=np.float32)
        valid_masks = np.array(valid_masks, dtype=bool)

        resized_heatmap_targets = []
        for heatmap_target_i in heatmap_targets:
            resized_heatmap_target_i = self.target_resize_function(
                heatmap_target_i, new_length
            )

            resized_heatmap_targets.append(resized_heatmap_target_i)
        resized_heatmap_targets = np.array(resized_heatmap_targets, dtype=np.float32)

        # remove series if they contain NaN values
        valid_series_index = [
            i
            for i, series_i in enumerate(resized_series)
            if not np.any(np.isnan(series_i))
        ]
        resized_series = resized_series[valid_series_index]
        valid_masks = valid_masks[valid_series_index]
        resized_heatmap_targets = resized_heatmap_targets[valid_series_index]
        index_targets = [index_targets[i] for i in valid_series_index]
        presence_targets = [presence_targets[i] for i in valid_series_index]
        original_shapes = [original_shapes[i] for i in valid_series_index]

        presence_targets = np.array(presence_targets, dtype=np.float32)

        if type(resized_series) is np.ndarray:
            resized_series = torch.tensor(resized_series, dtype=torch.float32)
        if type(resized_heatmap_targets) is np.ndarray:
            resized_heatmap_targets = torch.tensor(
                resized_heatmap_targets, dtype=torch.float32
            )
        if type(index_targets) is np.ndarray:
            index_targets = torch.tensor(index_targets, dtype=torch.float32)
        if type(presence_targets) is np.ndarray:
            presence_targets = torch.tensor(presence_targets, dtype=torch.float32)
        if type(valid_masks) is np.ndarray:
            valid_masks = torch.tensor(valid_masks, dtype=torch.bool)

        return (
            resized_series,
            valid_masks,
            resized_heatmap_targets,
            index_targets,
            presence_targets,
        )


class KeypointDetection1DPredictionDataset(Dataset):
    """
    PyTorch Dataset for 1D keypoint detection inference.

    Stores input time-series for prediction. The collate function pads or crops
    all series to the same length, builds a boolean mask marking the unpadded
    positions, and replaces NaN-containing series (and their masks) with zeros,
    returning their indices as ``invalid_series_index``.

    Parameters:
        inputs (array-like): Sequence of input series of shape ``(T,)`` or
            ``(C, T)``. 1D series are reshaped to ``(1, T)``.
        enforce_divisibility_by (int, optional): Target batch length is rounded
            to a multiple of this value. Pass ``None`` to skip resizing entirely.
            (default: 64)
        resize_method (str, optional): ``"pad"`` or ``"crop"``. (default: ``"pad"``)
    """

    def __init__(
        self,
        inputs,
        enforce_divisibility_by=64,
        resize_method="pad",
    ):
        self.input_series = inputs
        self.enforce_divisibility_by = enforce_divisibility_by

        self.resize_method = resize_method
        if resize_method == "pad":
            self.resize_function = pad_series_to_length
            self.multiplier_function = get_closest_upper_multiple
        elif resize_method == "crop":
            self.resize_function = crop_series_to_length
            self.multiplier_function = get_closest_lower_multiple

    def __len__(self):
        return len(self.input_series)

    def __getitem__(self, i):
        series = self.input_series[i]

        if series.ndim == 1:
            series = series.reshape(1, -1)

        return series.astype(np.float32), series.shape

    def collate_fn(self, batch):
        """
        Resize a batch of series to a common length and neutralize invalid ones.

        Parameters:
            batch (list[tuple]): Samples returned by ``__getitem__``.

        Returns:
            tuple: ``(series, valid_masks, invalid_series_index, original_shapes)``,
            where ``series`` has shape ``(B, C, new_length)``, ``valid_masks`` has
            shape ``(B, new_length)`` and is ``True`` on the unpadded positions,
            and ``invalid_series_index`` lists the batch positions of the series
            that contained NaN values and were zeroed out. When
            ``enforce_divisibility_by`` is ``None``, the untouched
            ``(series, original_shapes)`` tuple is returned instead.
        """
        series, original_shapes = zip(*batch)

        if self.enforce_divisibility_by is None:
            return series, original_shapes

        elif self.resize_method == "crop":
            target_length = min([shape[-1] for shape in original_shapes])
        else:
            target_length = max([shape[-1] for shape in original_shapes])

        new_length = self.multiplier_function(
            target_length, self.enforce_divisibility_by
        )

        # resize the images
        resized_series = []
        valid_masks = []
        for series_i, original_length in zip(
            series, [shape[-1] for shape in original_shapes]
        ):
            resized_series_i = self.resize_function(series_i, new_length)

            # get the mask of unpadded values (i.e., the original values before padding)
            mask_i = np.zeros(new_length, dtype=bool)
            mask_i[: min(original_length, new_length)] = True
            valid_masks.append(mask_i)
            resized_series.append(resized_series_i)

        resized_series = np.array(resized_series, dtype=np.float32)
        valid_masks = np.array(valid_masks, dtype=bool)

        # any series containing NaN would cause the batch to fail, so we replace them with zeros
        invalid_series_index = [
            i for i, series_i in enumerate(resized_series) if np.any(np.isnan(series_i))
        ]
        for i in invalid_series_index:
            resized_series[i] = np.zeros(new_length, dtype=np.float32)
            valid_masks[i] = np.zeros(new_length, dtype=bool)
        if type(resized_series) is np.ndarray:
            resized_series = torch.tensor(resized_series, dtype=torch.float32)
        if type(valid_masks) is np.ndarray:
            valid_masks = torch.tensor(valid_masks, dtype=torch.bool)

        return resized_series, valid_masks, invalid_series_index, original_shapes


def split_dataset(dataframe, validation_size, test_size):
    """
    Split a DataFrame (or CSV path) into training, validation, and test sets.

    Parameters:
        dataframe (pd.DataFrame or str): DataFrame or path to a CSV file.
        validation_size (float): Fraction of the total data for validation.
        test_size (float): Fraction of the total data for testing.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_dataframe, validation_dataframe, test_dataframe)``.

    Raises:
        ValueError: If ``validation_size + test_size >= 1.0``.
    """
    # Load the dataset
    if isinstance(dataframe, str):
        dataframe = pd.read_csv(dataframe)

    # Ensure the sizes are valid
    if validation_size + test_size >= 1.0:
        raise ValueError(
            "The sum of validation_size and test_size should be less than 1."
        )

    # Calculate the size of the temporary set (validation + test)
    temp_size = validation_size + test_size

    # Split the dataframe into training and temporary set
    train_dataframe, temp_dataframe = train_test_split(
        dataframe, test_size=temp_size, random_state=42
    )

    # Calculate the proportion of validation and test set relative to the temporary set
    validation_proportion = validation_size / temp_size

    # Split the temporary set into validation and test sets
    validation_dataframe, test_dataframe = train_test_split(
        temp_dataframe, test_size=1 - validation_proportion, random_state=42
    )

    return train_dataframe, validation_dataframe, test_dataframe


def create_segmentation_training_dataframes(
    image_directories,
    mask_directories,
    save_dir,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
):
    """
    Build training and validation DataFrames from image and mask directories.

    Pairs files by sorted order within each directory pair. Saves date-stamped
    CSV backups of all three splits to ``save_dir/database_backup/``.

    Parameters:
        image_directories (str or list[str]): Directories containing image files.
        mask_directories (str or list[str]): Directories containing mask files,
            paired with ``image_directories``.
        save_dir (str): Directory where backup CSVs are written.
        validation_set_ratio (float, optional): Fraction of data for validation.
            (default: 0.25)
        test_set_ratio (float, optional): Fraction of data for testing.
            (default: 0.1)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(training_dataframe, validation_dataframe)``.

    Raises:
        AssertionError: If the number of images and masks in a directory pair
            does not match.
    """
    if not isinstance(image_directories, list):
        image_directories = [image_directories]
    if not isinstance(mask_directories, list):
        mask_directories = [mask_directories]

    images = []
    masks = []
    for image_directory, mask_directory in zip(image_directories, mask_directories):
        images.extend(
            sorted(
                [
                    os.path.join(image_directory, file)
                    for file in os.listdir(image_directory)
                ]
            )
        )
        masks.extend(
            sorted(
                [
                    os.path.join(mask_directory, file)
                    for file in os.listdir(mask_directory)
                ]
            )
        )

    assert len(images) == len(
        masks
    ), "The number of images and masks in the directories must be equal"
    dataframe = pd.DataFrame({"image": images, "mask": masks})

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        dataframe, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    return training_dataframe, validation_dataframe


def get_unique_shapes_from_tiffs(
    image_paths: list[str], channels_to_keep: list[int] | None = None
) -> np.ndarray:
    """
    Get unique shapes from a list of TIFF images in parallel.

    Parameters:
        image_paths (list[str]): List of image paths to extract shapes from.
        channels_to_keep (list[int] | None): List of channel indices to keep. If ``None``, all channels are considered. (default: None)

    Returns:
        np.ndarray: Unique image shapes found in the dataframe
    """

    shapes = Parallel(n_jobs=-1)(
        delayed(image_handling.get_shape_from_tiff)(
            image_path, channels_to_keep=channels_to_keep
        )
        for image_path in image_paths
    )

    valid_shapes = [shape for shape in shapes if shape is not None]

    if len(valid_shapes) == 0:
        raise ValueError("No valid shapes found in the dataframe")

    all_shapes = set(valid_shapes)
    # because of potential rotation during augmentation, add the permutation
    for shape in valid_shapes:
        if len(shape) >= 2:
            rotated = shape[:-2] + (shape[-1], shape[-2])
            all_shapes.add(rotated)

    all_shapes = list(all_shapes)

    for i, shape in enumerate(all_shapes):
        if len(shape) == 3:
            # switch the axes to go from (C, H, W) to (H, W, C)
            shape = (shape[1], shape[2], shape[0])
            all_shapes[i] = shape
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
            all_shapes[i] = shape

    return np.array(all_shapes)


def create_segmentation_dataloaders(
    training_dataframe,
    validation_dataframe,
    channels,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    """
    Create training and validation DataLoaders for segmentation.

    When ``train_on_tiles`` is ``True``, uses :class:`TiledSegmentationDataset`
    with per-shape ``ImageSlicer`` objects; otherwise uses
    :class:`SegmentationDataset` with full images. Default transforms (percentile
    normalization) are applied if no transform is supplied.

    Parameters:
        training_dataframe (pd.DataFrame): DataFrame with ``"image"`` and
            ``"mask"`` columns for training data.
        validation_dataframe (pd.DataFrame): DataFrame with ``"image"`` and
            ``"mask"`` columns for validation data.
        channels (int or list[int]): Channel indices to load.
        batch_size (int, optional): Batch size. (default: 5)
        num_workers (int, optional): DataLoader worker processes. (default: 32)
        pin_memory (bool, optional): Whether to pin memory. (default: True)
        train_on_tiles (bool, optional): If ``True``, sample random tiles;
            otherwise use full images. (default: True)
        tiler_params (dict, optional): Required when ``train_on_tiles`` is
            ``True``; must contain ``"tile_size"`` and ``"tile_step"`` keys.
            (default: None)
        training_transform (callable, optional): Override transform for training.
            (default: None)
        validation_transform (callable, optional): Override transform for
            validation. (default: None)

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, val_loader)``.
    """
    if not train_on_tiles:
        train_loader = DataLoader(
            SegmentationDataset(
                training_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=training_transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            SegmentationDataset(
                validation_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=validation_transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    assert (
        tiler_params is not None
    ), "If train_on_tiles is True, tiler_params must be provided"

    image_paths = training_dataframe["image"].values.tolist()

    unique_shapes = get_unique_shapes_from_tiffs(image_paths, channels_to_keep=channels)

    image_slicers = {
        tuple(shape): inference.ImageSlicer(
            shape, tiler_params["tile_size"], tiler_params["tile_step"]
        )
        for shape in unique_shapes
    }

    if training_transform is None:
        training_transform = get_training_augmentation("percentile", lo=1, hi=99)
    if validation_transform is None:
        validation_transform = get_prediction_augmentation("percentile", lo=1, hi=99)

    train_loader = DataLoader(
        TiledSegmentationDataset(
            training_dataframe,
            image_slicers,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=training_transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TiledSegmentationDataset(
            validation_dataframe,
            image_slicers,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=validation_transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_segmentation_training_dataframes_and_dataloaders(
    image_directories,
    mask_directories,
    save_dir,
    channels,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    """
    Build training DataFrames and DataLoaders for segmentation in one step.

    Combines :func:`create_segmentation_training_dataframes` and
    :func:`create_segmentation_dataloaders`.

    Parameters:
        image_directories (str or list[str]): Directories containing image files.
        mask_directories (str or list[str]): Directories containing mask files.
        save_dir (str): Directory where backup CSVs are written.
        channels (int or list[int]): Channel indices to load.
        validation_set_ratio (float, optional): Fraction for validation.
            (default: 0.25)
        test_set_ratio (float, optional): Fraction for testing. (default: 0.1)
        batch_size (int, optional): Batch size. (default: 5)
        num_workers (int, optional): DataLoader worker processes. (default: 32)
        pin_memory (bool, optional): Whether to pin memory. (default: True)
        train_on_tiles (bool, optional): Whether to train on tiles. (default: True)
        tiler_params (dict, optional): Tile parameters; see
            :func:`create_segmentation_dataloaders`. (default: None)
        training_transform (callable, optional): Override training transform.
            (default: None)
        validation_transform (callable, optional): Override validation transform.
            (default: None)

    Returns:
        tuple: ``(training_dataframe, validation_dataframe, train_loader, val_loader)``.
    """
    (
        training_dataframe,
        validation_dataframe,
    ) = create_segmentation_training_dataframes(
        image_directories,
        mask_directories,
        save_dir,
        validation_set_ratio=validation_set_ratio,
        test_set_ratio=test_set_ratio,
    )
    train_loader, val_loader = create_segmentation_dataloaders(
        training_dataframe,
        validation_dataframe,
        channels,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=training_transform,
        validation_transform=validation_transform,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader


def create_segmentation_dataloaders_from_filemap(
    filemap_path,
    save_dir,
    channels,
    image_column="image",
    mask_column="mask",
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    """
    Build segmentation DataLoaders from a CSV filemap.

    Reads a CSV at ``filemap_path``, renames the specified columns to
    ``"image"`` and ``"mask"``, splits into train/val/test sets, saves
    date-stamped backups, and returns DataLoaders.

    Parameters:
        filemap_path (str): Path to the CSV filemap.
        save_dir (str): Directory where backup CSVs are written.
        channels (int or list[int]): Channel indices to load.
        image_column (str, optional): CSV column name for image paths.
            (default: ``"image"``)
        mask_column (str, optional): CSV column name for mask paths.
            (default: ``"mask"``)
        validation_set_ratio (float, optional): Fraction for validation.
            (default: 0.25)
        test_set_ratio (float, optional): Fraction for testing. (default: 0.1)
        batch_size (int, optional): Batch size. (default: 5)
        num_workers (int, optional): DataLoader worker processes. (default: 32)
        pin_memory (bool, optional): Whether to pin memory. (default: True)
        train_on_tiles (bool, optional): Whether to train on tiles. (default: True)
        tiler_params (dict, optional): Tile parameters; see
            :func:`create_segmentation_dataloaders`. (default: None)
        training_transform (callable, optional): Override training transform.
            (default: None)
        validation_transform (callable, optional): Override validation transform.
            (default: None)

    Returns:
        tuple: ``(training_dataframe, validation_dataframe, train_loader, val_loader)``.
    """
    dataframe = pd.read_csv(filemap_path)
    # rename image column to "image" and mask column to "mask"
    dataframe = dataframe.rename(columns={image_column: "image", mask_column: "mask"})

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        dataframe, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    train_loader, val_loader = create_segmentation_dataloaders(
        training_dataframe,
        validation_dataframe,
        channels,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=training_transform,
        validation_transform=validation_transform,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader


def create_classification_training_dataframes(
    ground_truth_csv_paths,
    image_columns,
    class_columns,
    save_dir,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
):
    """
    Build training and validation DataFrames for classification from CSV ground-truth files.

    Reads one or more CSV files, extracts image path and class label columns,
    concatenates them, splits into train/val/test sets, and saves date-stamped
    CSV backups.

    Parameters:
        ground_truth_csv_paths (str or list[str]): Paths to ground-truth CSV files.
        image_columns (str or list[str]): Column name(s) for image paths. A single
            string is broadcast to all CSVs.
        class_columns (str or list[str]): Column name(s) for class labels.
        save_dir (str): Directory where backup CSVs are written.
        validation_set_ratio (float, optional): Fraction for validation.
            (default: 0.25)
        test_set_ratio (float, optional): Fraction for testing. (default: 0.1)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(training_dataframe, validation_dataframe)``.
    """
    if not isinstance(ground_truth_csv_paths, list):
        ground_truth_csv_paths = [ground_truth_csv_paths]
    if isinstance(image_columns, str):
        image_columns = [image_columns]
    if isinstance(class_columns, str):
        class_columns = [class_columns]

    if len(image_columns) == 1:
        image_columns = image_columns * len(ground_truth_csv_paths)
    if len(class_columns) == 1:
        class_columns = class_columns * len(ground_truth_csv_paths)

    ground_truth_df = pd.DataFrame()
    for i, ground_truth_csv in enumerate(ground_truth_csv_paths):
        gt_df = pd.read_csv(ground_truth_csv)
        images = gt_df[image_columns[i]].values.tolist()
        classes = gt_df[class_columns[i]].values.tolist()
        new_gt_df = pd.DataFrame({"image": images, "class": classes})
        ground_truth_df = pd.concat([ground_truth_df, new_gt_df], ignore_index=True)

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        ground_truth_df, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    return training_dataframe, validation_dataframe


def create_classification_dataloaders(
    training_dataframe,
    validation_dataframe,
    channels,
    n_classes,
    batch_size=64,
    num_workers=32,
    pin_memory=True,
    training_transform=None,
    validation_transform=None,
):
    """
    Create training and validation DataLoaders for image classification.

    Parameters:
        training_dataframe (pd.DataFrame): DataFrame with ``"image"`` and
            ``"class"`` columns for training data.
        validation_dataframe (pd.DataFrame): DataFrame with ``"image"`` and
            ``"class"`` columns for validation data.
        channels (int or list[int]): Channel indices to load.
        n_classes (int): Number of classes (labels are one-hot encoded when > 2).
        batch_size (int, optional): Batch size. (default: 64)
        num_workers (int, optional): DataLoader worker processes. (default: 32)
        pin_memory (bool, optional): Whether to pin memory. (default: True)
        training_transform (callable, optional): Override training transform.
            (default: None)
        validation_transform (callable, optional): Override validation transform.
            (default: None)

    Returns:
        tuple[DataLoader, DataLoader]: ``(train_loader, val_loader)``.
    """
    train_loader = DataLoader(
        ClassificationDataset(
            training_dataframe,
            channels=channels,
            n_classes=n_classes,
            class_column="class",
            image_column="image",
            transform=training_transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ClassificationDataset(
            validation_dataframe,
            channels=channels,
            n_classes=n_classes,
            class_column="class",
            image_column="image",
            transform=validation_transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader

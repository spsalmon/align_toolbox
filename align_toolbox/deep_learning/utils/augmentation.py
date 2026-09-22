import numpy as np
from csbdeep.utils import normalize
from monai.data import set_track_meta
from monai.transforms import (
    Compose,
    MapTransform,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    Randomizable,
    Transform,
)

from align_toolbox.foundation import image_handling

# Avoid MetaTensor overhead when working with plain numpy arrays
set_track_meta(False)


# ---------------------------------------------------------------------------
# Intensity transforms
# ---------------------------------------------------------------------------


class NormalizeDataRange(MapTransform):
    """
    MONAI MapTransform that normalizes arrays to [0, 1] by their per-sample min/max.

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
    """

    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = (d[key] - d[key].min()) / (d[key].max() - d[key].min())
        return d


class NormalizeMeanStd(MapTransform):
    """
    MONAI MapTransform that standardizes arrays using a fixed mean and std.

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
        mean (float): Mean to subtract.
        std (float): Standard deviation to divide by.
    """

    def __init__(self, keys, mean: float, std: float):
        MapTransform.__init__(self, keys)
        self.mean = mean
        self.std = std

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = (d[key] - self.mean) / self.std
        return d


class NormalizePercentile(MapTransform):
    """
    MONAI MapTransform that normalizes arrays using percentile clipping (csbdeep).

    Clips values at the ``lo``-th and ``hi``-th percentiles and rescales to [0, 1].

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
        lo (float): Lower percentile for clipping (e.g. 1 for the 1st percentile).
        hi (float): Upper percentile for clipping (e.g. 99 for the 99th percentile).
        axis (int or None, optional): Axis over which to compute percentiles.
            ``None`` uses the global min/max. (default: None)
    """

    def __init__(self, keys, lo: float, hi: float, axis=None):
        MapTransform.__init__(self, keys)
        self.lo = lo
        self.hi = hi
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = normalize(d[key], self.lo, self.hi, axis=self.axis)
        return d


class EnforceNChannels(MapTransform):
    """
    MONAI MapTransform that tiles channel data to reach exactly ``n_channels``.

    Delegates to :func:`_enforce_n_channels` for the actual tiling logic.

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
        n_channels (int): Target number of channels.
    """

    def __init__(self, keys, n_channels: int):
        MapTransform.__init__(self, keys)
        self.n_channels = n_channels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = _enforce_n_channels(d[key], self.n_channels)
        return d


# ---------------------------------------------------------------------------
# Geometric transforms (image + mask)
# ---------------------------------------------------------------------------


class CustomFlip(MapTransform, Randomizable):
    """
    MONAI Randomizable MapTransform that randomly flips arrays along spatial axes.

    With probability ``prob``, flips along one of: height axis only, width axis
    only, or both height and width axes.

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
        prob (float, optional): Probability of applying the flip. (default: 0.75)
    """

    _FLIP_OPTIONS = [(-2,), (-1,), (-1, -2)]

    def __init__(self, keys, prob=0.75):
        MapTransform.__init__(self, keys)
        self.prob = prob
        self._do_transform = False
        self._axes = None

    def randomize(self, data=None):
        self._do_transform = self.R.random() < self.prob
        self._axes = self._FLIP_OPTIONS[self.R.randint(3)]

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = np.flip(d[key], axis=self._axes)
        return d


class CustomRotate90(MapTransform, Randomizable):
    """
    MONAI Randomizable MapTransform that randomly rotates arrays by 90°, 180°, or 270°.

    With probability ``prob``, rotates in the spatial (H, W) plane by a randomly
    chosen multiple of 90°.

    Parameters:
        keys (sequence): Keys in the data dictionary to apply the transform to.
        prob (float, optional): Probability of applying the rotation. (default: 0.75)
    """

    def __init__(self, keys, prob=0.75):
        MapTransform.__init__(self, keys)
        self.prob = prob
        self._do_transform = False
        self._k = None

    def randomize(self, data=None):
        self._do_transform = self.R.random() < self.prob
        self._k = self.R.choice([1, 2, 3])

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = np.rot90(d[key], k=self._k, axes=(-2, -1))
        return d


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_normalization(keys, normalization_type: str, **kwargs) -> Transform:
    """
    Instantiate the appropriate normalization transform for the given type.

    Parameters:
        keys (sequence): Keys in the data dictionary to normalize.
        normalization_type (str): One of ``"data_range"``, ``"mean_std"``,
            ``"percentile"``.
        **kwargs: Parameters forwarded to the chosen transform constructor
            (e.g. ``mean`` and ``std`` for ``"mean_std"``; ``lo`` and ``hi``
            for ``"percentile"``).

    Returns:
        Transform: Configured MONAI transform instance.

    Raises:
        ValueError: If ``normalization_type`` is not recognized.
    """
    if normalization_type == "data_range":
        return NormalizeDataRange(keys)
    elif normalization_type == "mean_std":
        return NormalizeMeanStd(keys, kwargs["mean"], kwargs["std"])
    elif normalization_type == "percentile":
        return NormalizePercentile(keys, kwargs["lo"], kwargs["hi"], kwargs.get("axis"))
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def get_training_augmentation(normalization_type: str, **kwargs) -> Compose:
    """
    Build the MONAI augmentation pipeline for segmentation training.

    Includes random flips, random 90° rotations, random Gaussian smoothing,
    random Gaussian sharpening, and normalization.

    Parameters:
        normalization_type (str): Normalization type passed to
            :func:`_build_normalization` (``"data_range"``, ``"mean_std"``,
            ``"percentile"``).
        **kwargs: Additional parameters forwarded to the normalization transform
            and optionally ``enforce_n_channels`` (int) to tile channels.

    Returns:
        Compose: MONAI Compose pipeline ready for training.
    """
    transforms = [
        CustomFlip(keys=["image", "mask"], prob=0.75),
        CustomRotate90(keys=["image", "mask"], prob=0.75),
        RandGaussianSmoothd(
            keys=["image"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)
        ),
        RandGaussianSharpend(keys=["image"], prob=0.5),
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        ),
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(keys=["image"], n_channels=n))

    return Compose(transforms)


def get_qc_training_augmentation(normalization_type: str, **kwargs) -> Compose:
    """
    Build the MONAI augmentation pipeline for quality-control model training.

    Lighter than :func:`get_training_augmentation`: includes only random flips
    and normalization (no rotations or intensity transforms).

    Parameters:
        normalization_type (str): Normalization type passed to
            :func:`_build_normalization`.
        **kwargs: Additional parameters forwarded to the normalization transform
            and optionally ``enforce_n_channels`` (int).

    Returns:
        Compose: MONAI Compose pipeline ready for QC training.
    """
    transforms = [
        CustomFlip(keys=["image"], prob=0.75),
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        ),
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(keys=["image"], n_channels=n))

    return Compose(transforms)


def get_prediction_augmentation(normalization_type: str, **kwargs) -> Compose:
    """
    Build the MONAI transform pipeline for inference (normalization only).

    Parameters:
        normalization_type (str): Normalization type passed to
            :func:`_build_normalization`.
        **kwargs: Additional parameters forwarded to the normalization transform
            and optionally ``enforce_n_channels`` (int).

    Returns:
        Compose: MONAI Compose pipeline ready for prediction.
    """
    transforms = [
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        )
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(keys=["image"], n_channels=n))

    return Compose(transforms)


def get_prediction_augmentation_from_model(model, enforce_n_channels=None) -> Compose:
    """
    Build the inference transform pipeline from a model's stored normalization config.

    Reads the ``normalization`` attribute of ``model`` (a dict with at least a
    ``"type"`` key) and delegates to :func:`get_prediction_augmentation`.

    Parameters:
        model: A model instance exposing a ``normalization`` dict attribute
            (e.g. a ``PretrainedSegmentationModel``).
        enforce_n_channels (int, optional): If not ``None``, tile channels to this
            count via :class:`EnforceNChannels`. (default: None)

    Returns:
        Compose: MONAI Compose pipeline ready for prediction.
    """
    params = model.normalization
    return get_prediction_augmentation(
        normalization_type=params["type"],
        enforce_n_channels=enforce_n_channels,
        **{k: v for k, v in params.items() if k != "type"},
    )


def get_mean_and_std(image_path: str) -> tuple[float, float]:
    """
    Compute the mean and standard deviation of channel 2 in a TIFF image.

    Parameters:
        image_path (str): Path to the TIFF image file.

    Returns:
        tuple[float, float]: ``(mean, std)`` of the pixel values in channel 2.
    """
    image = image_handling.read_tiff_file(image_path, [2])
    return float(np.mean(image)), float(np.std(image))


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _enforce_n_channels(image: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Tile image channels to reach exactly ``n_channels``.

    If the image has fewer channels than ``n_channels``, tiles the existing
    channels (distributing any remainder to the first channels).

    Parameters:
        image (np.ndarray): Image array of shape ``(H, W)`` or ``(C, H, W)``.
        n_channels (int): Desired number of output channels.

    Returns:
        np.ndarray: Array of shape ``(n_channels, H, W)``.

    Raises:
        AssertionError: If ``image`` has more than 3 dimensions.
        ValueError: If the image already has more channels than ``n_channels``.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)

    assert image.ndim <= 3, "Multichannel z-stacks are not supported"

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    c = image.shape[0]
    if c == n_channels:
        return image
    if c > n_channels:
        raise ValueError(f"Image has {c} channels, expected at most {n_channels}")

    base = n_channels // c
    remainder = n_channels % c

    repeated = np.tile(image, (base, 1, 1))
    if remainder == 0:
        return repeated
    return np.concatenate([repeated, image[:remainder]], axis=0)

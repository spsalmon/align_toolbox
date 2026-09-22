import os

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from tifffile import imwrite

from align_toolbox.deep_learning.utils import dataset as ds


@pytest.fixture(autouse=True)
def _threading_backend():
    """
    get_unique_shapes_from_tiffs hard-codes n_jobs=-1; loky would spawn one
    interpreter per core, each re-importing torch and the package.
    """
    with joblib.parallel_config(backend="threading"):
        yield


def _write(path, array):
    imwrite(path, array)
    return str(path)


@pytest.fixture
def image_mask_pairs(tmp_path):
    """Four (C=2, H, W) images with matching masks of varying sizes."""
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rows = []
    for i, (height, width) in enumerate([(40, 50), (40, 50), (30, 70), (60, 60)]):
        image = (rng.random((2, height, width)) * 1000).astype(np.uint16)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[5:15, 5:25] = 1
        rows.append(
            {
                "image": _write(tmp_path / "images" / f"img_{i}.tiff", image),
                "mask": _write(tmp_path / "masks" / f"img_{i}.tiff", mask),
            }
        )
    return pd.DataFrame(rows)


# --- segmentation ---------------------------------------------------------------


def test_segmentation_dataset_returns_channel_first_image_and_mask(image_mask_pairs):
    dataset = ds.SegmentationDataset(image_mask_pairs, channels=1)
    image, mask = dataset[0]
    assert len(dataset) == 4
    assert image.shape == (1, 40, 50) and image.dtype == np.float32
    assert mask.shape == (1, 40, 50)


def test_segmentation_dataset_rejects_unknown_resize_mode(image_mask_pairs):
    with pytest.raises(ValueError):
        ds.SegmentationDataset(image_mask_pairs, channels=0, pad_or_crop="stretch")


def test_segmentation_dataset_collate_pads_to_common_multiple(image_mask_pairs):
    dataset = ds.SegmentationDataset(image_mask_pairs, channels=[0, 1])
    images, masks = dataset.collate_fn([dataset[0], dataset[2]])
    assert images.shape == (2, 2, 64, 96)
    assert masks.shape == (2, 1, 64, 96)


def test_tiled_segmentation_dataset_returns_random_tile(image_mask_pairs):
    from pytorch_toolbelt import inference

    shapes = ds.get_unique_shapes_from_tiffs(
        image_mask_pairs["image"].tolist(), channels_to_keep=[0, 1]
    )
    slicers = {tuple(s): inference.ImageSlicer(s, 16, 8) for s in shapes}
    dataset = ds.TiledSegmentationDataset(image_mask_pairs, slicers, channels=[0, 1])
    image, mask = dataset[2]
    assert image.shape == (2, 16, 16) and image.dtype == np.float32
    assert mask.shape == (1, 16, 16)


def test_get_unique_shapes_from_tiffs_adds_rotations_channel_last(image_mask_pairs):
    shapes = ds.get_unique_shapes_from_tiffs(image_mask_pairs["image"].tolist())
    assert {tuple(s) for s in shapes} == {
        (40, 50, 2),
        (50, 40, 2),
        (30, 70, 2),
        (70, 30, 2),
        (60, 60, 2),
    }


def test_get_unique_shapes_from_tiffs_single_channel(image_mask_pairs):
    shapes = ds.get_unique_shapes_from_tiffs(
        image_mask_pairs["mask"].tolist()[:1], channels_to_keep=None
    )
    assert {tuple(s) for s in shapes} == {(40, 50, 1), (50, 40, 1)}


def test_get_unique_shapes_from_tiffs_without_valid_files_raises(tmp_path):
    with pytest.raises(ValueError):
        ds.get_unique_shapes_from_tiffs([str(tmp_path / "missing.tiff")])


def test_get_unique_shapes_from_tiffs_requires_image_paths():
    """
    image_paths used to carry ``list[str]`` as a default value instead of an
    annotation, so a call with no paths iterated the generic alias and failed
    with a misleading "no valid shapes" error.
    """
    with pytest.raises(TypeError):
        ds.get_unique_shapes_from_tiffs()


@pytest.mark.parametrize(
    "pad_or_crop, expected_shape", [("pad", (2, 1, 64, 96)), ("crop", (2, 1, 0, 32))]
)
def test_segmentation_prediction_collate(image_mask_pairs, pad_or_crop, expected_shape):
    dataset = ds.SegmentationPredictionDataset(
        image_mask_pairs["image"].tolist(), channels=0, pad_or_crop=pad_or_crop
    )
    paths, images, shapes, invalid = dataset.collate_fn([dataset[0], dataset[2]])
    assert paths == image_mask_pairs["image"].tolist()[0:3:2]
    assert images.shape == expected_shape
    assert shapes == [(1, 40, 50), (1, 30, 70)]
    assert invalid == []


def test_segmentation_prediction_dataset_reports_unreadable_images(
    image_mask_pairs, tmp_path
):
    paths = [str(tmp_path / "missing.tiff")] + image_mask_pairs["image"].tolist()[:1]
    dataset = ds.SegmentationPredictionDataset(paths, channels=0)
    assert dataset[0] == (paths[0], None, None)
    returned_paths, images, _, invalid = dataset.collate_fn([dataset[0], dataset[1]])
    assert invalid == [0]
    assert returned_paths == paths[1:]
    assert images.shape == (1, 1, 64, 64)
    assert dataset.collate_fn([dataset[0]]) == (None, None, None, None)


def test_segmentation_prediction_dataset_rescales_and_transforms(image_mask_pairs):
    from align_toolbox.deep_learning.utils.augmentation import (
        get_prediction_augmentation,
    )

    dataset = ds.SegmentationPredictionDataset(
        image_mask_pairs["image"].tolist(),
        channels=0,
        scale_factor=0.5,
        transform=get_prediction_augmentation("data_range"),
    )
    _, image, shape = dataset[0]
    assert shape == (1, 20, 25)
    assert image.min() >= 0 and image.max() <= 1


def test_segmentation_prediction_dataset_rejects_unknown_resize_mode():
    with pytest.raises(ValueError):
        ds.SegmentationPredictionDataset([], channels=0, pad_or_crop="stretch")


@pytest.mark.parametrize(
    "stack_shape, kwargs, expected_plane_shape",
    [
        ((3, 50, 70), {}, (1, 64, 96)),
        ((3, 50, 70), {"pad_or_crop": "crop"}, (1, 32, 64)),
        ((3, 50, 70), {"scale_factor": 0.5}, (1, 32, 64)),
        ((3, 2, 50, 70), {"scale_factor": 0.5}, (2, 32, 64)),
    ],
)
def test_stack_prediction_dataset_planes(stack_shape, kwargs, expected_plane_shape):
    stack = np.ones(stack_shape, dtype=np.float32)
    dataset = ds.StackPredictionDataset(stack, channels=None, **kwargs)
    assert len(dataset) == 3
    assert dataset[1].shape == expected_plane_shape
    assert dataset[1].dtype == np.float32


def test_stack_prediction_dataset_reads_path_and_applies_transform(tmp_path):
    from align_toolbox.deep_learning.utils.augmentation import (
        get_prediction_augmentation,
    )

    stack = np.arange(3 * 2 * 32 * 32, dtype=np.uint16).reshape(3, 2, 32, 32)
    path = _write(tmp_path / "stack.tiff", stack)
    dataset = ds.StackPredictionDataset(
        path, channels=1, transform=get_prediction_augmentation("data_range")
    )
    plane = dataset[0]
    assert plane.shape == (1, 32, 32)
    assert plane.min() == 0 and plane.max() == 1


def test_stack_prediction_dataset_rejects_unknown_resize_mode():
    with pytest.raises(ValueError):
        ds.StackPredictionDataset(np.ones((1, 8, 8)), None, pad_or_crop="stretch")


# --- classification and quality control ------------------------------------------


def test_classification_dataset_binary_labels(image_mask_pairs):
    dataframe = image_mask_pairs.assign(**{"class": [0, 1, 0, 1]})
    dataset = ds.ClassificationDataset(dataframe, channels=1, n_classes=2)
    image, label = dataset[1]
    assert image.shape == (1, 40, 50)
    assert label == 1.0


def test_classification_dataset_one_hot_encodes_multiclass(image_mask_pairs):
    dataframe = image_mask_pairs.assign(**{"class": [0, 1, 2, 1]})
    dataset = ds.ClassificationDataset(dataframe, channels=1, n_classes=3)
    np.testing.assert_array_equal(dataset[2][1], [0, 0, 1])


def test_quality_control_dataset_image_only(image_mask_pairs):
    dataset = ds.QualityControlDataset(
        image_mask_pairs["image"].tolist(),
        None,
        channels=0,
        labels=["good", "bad", "good", "bad"],
        classes=["good", "bad"],
    )
    image, label = dataset[1]
    assert image.shape == (1, 40, 50)
    assert label == 1


def test_quality_control_dataset_stacks_mask_as_last_channel(image_mask_pairs):
    dataset = ds.QualityControlDataset(
        image_mask_pairs["image"].tolist(),
        image_mask_pairs["mask"].tolist(),
        channels=[0, 1],
        labels=[0, 1, 0, 1],
        classes=["good", "bad"],
    )
    combined, label = dataset[0]
    assert combined.shape == (3, 40, 50)
    np.testing.assert_array_equal(combined[-1][5:15, 5:25], 1)
    assert label == 0


@pytest.mark.parametrize(
    "resize_method, expected_shape", [("pad", (3, 2, 64, 96)), ("crop", (3, 2, 32, 64))]
)
def test_quality_control_collate_drops_empty_masks(
    image_mask_pairs, tmp_path, resize_method, expected_shape
):
    empty_mask = _write(tmp_path / "empty.tiff", np.zeros((40, 50), dtype=np.uint8))
    masks = image_mask_pairs["mask"].tolist()
    masks[1] = empty_mask
    dataset = ds.QualityControlDataset(
        image_mask_pairs["image"].tolist(),
        masks,
        channels=0,
        labels=[0, 1, 1, 0],
        classes=["good", "bad"],
        resize_method=resize_method,
    )
    images, labels = dataset.collate_fn([dataset[i] for i in range(4)])
    assert images.shape == expected_shape
    assert labels.tolist() == [0, 1, 0]


def test_quality_control_collate_returns_none_for_single_valid_sample(
    image_mask_pairs, tmp_path
):
    empty_mask = _write(tmp_path / "empty.tiff", np.zeros((40, 50), dtype=np.uint8))
    dataset = ds.QualityControlDataset(
        image_mask_pairs["image"].tolist()[:2],
        [image_mask_pairs["mask"][0], empty_mask],
        channels=0,
        labels=[0, 1],
        classes=["good", "bad"],
    )
    assert dataset.collate_fn([dataset[0], dataset[1]]) is None


def test_quality_control_prediction_dataset_getitem(image_mask_pairs, tmp_path):
    images = image_mask_pairs["image"].tolist()
    masks = image_mask_pairs["mask"].tolist()
    dataset = ds.QualityControlPredictionDataset(
        images + [str(tmp_path / "missing.tiff")], masks + [masks[0]], channels=0
    )
    assert dataset[0].shape == (2, 40, 50)
    assert dataset[4] is None


def test_quality_control_prediction_collate_reports_rejected(
    image_mask_pairs, tmp_path
):
    empty_mask = _write(tmp_path / "empty.tiff", np.zeros((40, 50), dtype=np.uint8))
    dataset = ds.QualityControlPredictionDataset(
        image_mask_pairs["image"].tolist()[:2],
        [image_mask_pairs["mask"][0], empty_mask],
        channels=0,
    )
    images, rejected = dataset.collate_fn([dataset[0], dataset[1]])
    assert images.shape == (1, 2, 64, 64)
    assert rejected == [1]


# --- 1D keypoint detection -------------------------------------------------------


@pytest.fixture
def keypoint_samples():
    inputs = [np.arange(50, dtype=float), np.arange(70, dtype=float)]
    heatmaps = [np.zeros((2, 50)), np.zeros((2, 70))]
    indices = [np.array([10.0, np.nan]), np.array([5.0, 60.0])]
    return inputs, heatmaps, indices


def test_keypoint_training_dataset_getitem(keypoint_samples):
    dataset = ds.KeypointDetection1DTrainingDataset(*keypoint_samples)
    series, heatmap, index, presence, shape = dataset[0]
    assert series.shape == (1, 50) and series.dtype == np.float32
    assert heatmap.shape == (2, 50)
    assert index.shape == (1, 2)
    np.testing.assert_array_equal(presence, [1, 0])
    assert shape == (1, 50)


def test_keypoint_training_collate_pads_and_masks(keypoint_samples):
    dataset = ds.KeypointDetection1DTrainingDataset(*keypoint_samples)
    series, masks, heatmaps, indices, presence = dataset.collate_fn(
        [dataset[0], dataset[1]]
    )
    assert series.shape == (2, 1, 128)
    assert masks.dtype == torch.bool
    assert masks.sum(dim=1).tolist() == [50, 70]
    assert heatmaps.shape == (2, 2, 128)
    assert presence.tolist() == [[1, 0], [1, 1]]
    assert len(indices) == 2


def test_keypoint_training_collate_crop_mode(keypoint_samples):
    dataset = ds.KeypointDetection1DTrainingDataset(
        *keypoint_samples, enforce_divisibility_by=16, resize_method="crop"
    )
    series, masks, heatmaps, _, _ = dataset.collate_fn([dataset[0], dataset[1]])
    assert series.shape == (2, 1, 48)
    assert masks.all()


def test_keypoint_training_collate_drops_nan_series(keypoint_samples):
    inputs, heatmaps, indices = keypoint_samples
    inputs[1] = inputs[1].copy()
    inputs[1][3] = np.nan
    dataset = ds.KeypointDetection1DTrainingDataset(inputs, heatmaps, indices)
    series, masks, heatmaps, indices, presence = dataset.collate_fn(
        [dataset[0], dataset[1]]
    )
    assert series.shape[0] == masks.shape[0] == heatmaps.shape[0] == 1
    assert presence.tolist() == [[1, 0]]


def test_keypoint_training_collate_without_divisibility_is_passthrough(
    keypoint_samples,
):
    dataset = ds.KeypointDetection1DTrainingDataset(
        *keypoint_samples, enforce_divisibility_by=None
    )
    series, heatmaps, indices, shapes = dataset.collate_fn([dataset[0], dataset[1]])
    assert shapes == ((1, 50), (1, 70))


def test_keypoint_prediction_collate_zeroes_nan_series(keypoint_samples):
    inputs = keypoint_samples[0]
    inputs[0] = inputs[0].copy()
    inputs[0][0] = np.nan
    dataset = ds.KeypointDetection1DPredictionDataset(inputs)
    series, masks, invalid, shapes = dataset.collate_fn([dataset[0], dataset[1]])
    assert series.shape == (2, 1, 128)
    assert invalid == [0]
    assert not series[0].any() and not masks[0].any()
    assert masks[1].sum() == 70
    assert shapes == ((1, 50), (1, 70))


def test_keypoint_prediction_collate_crop_and_passthrough(keypoint_samples):
    inputs = keypoint_samples[0]
    cropped = ds.KeypointDetection1DPredictionDataset(
        inputs, enforce_divisibility_by=16, resize_method="crop"
    )
    series, _, _, _ = cropped.collate_fn([cropped[0], cropped[1]])
    assert series.shape == (2, 1, 48)
    raw = ds.KeypointDetection1DPredictionDataset(inputs, enforce_divisibility_by=None)
    assert raw.collate_fn([raw[0]])[1] == ((1, 50),)


# --- dataframe and dataloader factories -------------------------------------------


def test_split_dataset_proportions_are_deterministic():
    dataframe = pd.DataFrame({"x": range(100)})
    train, validation, test = ds.split_dataset(dataframe, 0.25, 0.25)
    assert (len(train), len(validation), len(test)) == (50, 25, 25)
    assert set(train.x) | set(validation.x) | set(test.x) == set(range(100))
    again = ds.split_dataset(dataframe, 0.25, 0.25)
    assert train.equals(again[0])


def test_split_dataset_reads_csv_and_validates_sizes(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(10)}).to_csv(path, index=False)
    assert sum(len(part) for part in ds.split_dataset(str(path), 0.2, 0.2)) == 10
    with pytest.raises(ValueError):
        ds.split_dataset(str(path), 0.5, 0.5)


def _backup_files(save_dir):
    return sorted(
        name.rsplit("_", 1)[0] for name in os.listdir(save_dir / "database_backup")
    )


def test_create_segmentation_training_dataframes(image_mask_pairs, tmp_path):
    save_dir = tmp_path / "run"
    train, validation = ds.create_segmentation_training_dataframes(
        str(tmp_path / "images"), str(tmp_path / "masks"), str(save_dir), 0.25, 0.25
    )
    assert len(train) + len(validation) == 3
    for _, row in pd.concat([train, validation]).iterrows():
        assert os.path.basename(row["image"]) == os.path.basename(row["mask"])
    assert _backup_files(save_dir) == [
        "test_dataframe",
        "training_dataframe",
        "validation_dataframe",
    ]


def test_create_segmentation_training_dataframes_count_mismatch(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.tiff").write_bytes(b"")
    with pytest.raises(AssertionError):
        ds.create_segmentation_training_dataframes(
            str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path)
        )


def test_create_segmentation_dataloaders_full_images(image_mask_pairs):
    train_loader, val_loader = ds.create_segmentation_dataloaders(
        image_mask_pairs,
        image_mask_pairs,
        channels=0,
        num_workers=0,
        batch_size=1,
        train_on_tiles=False,
    )
    images, masks = next(iter(val_loader))
    assert isinstance(train_loader.dataset, ds.SegmentationDataset)
    assert images.shape == (1, 1, 40, 50)


def test_create_segmentation_dataloaders_on_tiles(image_mask_pairs):
    train_loader, val_loader = ds.create_segmentation_dataloaders(
        image_mask_pairs,
        image_mask_pairs,
        channels=[0, 1],
        num_workers=0,
        batch_size=2,
        pin_memory=False,
        tiler_params={"tile_size": 16, "tile_step": 8},
    )
    images, masks = next(iter(train_loader))
    assert images.shape == (2, 2, 16, 16)
    assert masks.shape == (2, 1, 16, 16)
    # the default validation transform applies percentile normalization
    images, _ = next(iter(val_loader))
    assert images.abs().max() < 5


def test_create_segmentation_dataloaders_on_tiles_requires_params(image_mask_pairs):
    with pytest.raises(AssertionError):
        ds.create_segmentation_dataloaders(image_mask_pairs, image_mask_pairs, 0)


def test_create_segmentation_training_dataframes_and_dataloaders(
    image_mask_pairs, tmp_path
):
    (
        train,
        validation,
        train_loader,
        val_loader,
    ) = ds.create_segmentation_training_dataframes_and_dataloaders(
        str(tmp_path / "images"),
        str(tmp_path / "masks"),
        str(tmp_path / "run"),
        channels=0,
        validation_set_ratio=0.25,
        test_set_ratio=0.25,
        num_workers=0,
        train_on_tiles=False,
    )
    assert len(train_loader.dataset) == len(train)
    assert len(val_loader.dataset) == len(validation)


def test_create_segmentation_dataloaders_from_filemap_renames_columns(
    image_mask_pairs, tmp_path
):
    filemap_path = tmp_path / "filemap.csv"
    image_mask_pairs.rename(columns={"image": "raw", "mask": "seg"}).to_csv(
        filemap_path, index=False
    )
    train, _, train_loader, _ = ds.create_segmentation_dataloaders_from_filemap(
        str(filemap_path),
        str(tmp_path / "run"),
        channels=0,
        image_column="raw",
        mask_column="seg",
        validation_set_ratio=0.25,
        test_set_ratio=0.25,
        num_workers=0,
        train_on_tiles=False,
    )
    assert "image" in train.columns


def test_create_segmentation_dataloaders_from_filemap(image_mask_pairs, tmp_path):
    filemap_path = tmp_path / "filemap.csv"
    image_mask_pairs.to_csv(filemap_path, index=False)
    save_dir = tmp_path / "run"
    (
        train,
        validation,
        train_loader,
        _,
    ) = ds.create_segmentation_dataloaders_from_filemap(
        str(filemap_path),
        str(save_dir),
        channels=0,
        validation_set_ratio=0.25,
        test_set_ratio=0.25,
        num_workers=0,
        train_on_tiles=False,
    )
    assert len(train) + len(validation) == 3
    assert len(_backup_files(save_dir)) == 3


@pytest.fixture
def ground_truth_csvs(image_mask_pairs, tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"ground_truth_{i}.csv"
        pd.DataFrame({"path": image_mask_pairs["image"], "label": [0, 1, 0, 1]}).to_csv(
            path, index=False
        )
        paths.append(str(path))
    return paths


def test_create_classification_training_dataframes_single_csv(
    ground_truth_csvs, tmp_path
):
    train, validation = ds.create_classification_training_dataframes(
        ground_truth_csvs[0], "path", "label", str(tmp_path / "run"), 0.25, 0.25
    )
    assert list(train.columns) == ["image", "class"]
    assert len(train) + len(validation) == 3


def test_create_classification_training_dataframes_broadcasts_columns(
    ground_truth_csvs, tmp_path
):
    train, validation = ds.create_classification_training_dataframes(
        ground_truth_csvs, "path", "label", str(tmp_path / "run"), 0.25, 0.25
    )
    assert len(train) + len(validation) == 6


def test_create_classification_dataloaders(image_mask_pairs):
    dataframe = image_mask_pairs.assign(**{"class": [0, 1, 0, 1]})
    train_loader, val_loader = ds.create_classification_dataloaders(
        dataframe, dataframe, channels=0, n_classes=2, batch_size=2, num_workers=0
    )
    images, labels = next(iter(val_loader))
    assert images.shape == (2, 1, 40, 50)
    assert labels.tolist() == [0.0, 1.0]
    assert train_loader.dataset.ground_truth == [0.0, 1.0, 0.0, 1.0]

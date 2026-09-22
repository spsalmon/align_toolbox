import pytest
import pytorch_lightning as pl
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from align_toolbox.deep_learning import deep_learning_tools as dlt
from align_toolbox.deep_learning.architectures import models
from align_toolbox.deep_learning.utils.dataset import KeypointDetection1DTrainingDataset
from align_toolbox.deep_learning.utils.loss import FocalTverskyLoss

SEGMENTATION_KWARGS = dict(
    architecture="Unet", encoder="resnet18", pretrained_weights=None
)


@pytest.fixture(autouse=True)
def _no_weight_downloads(monkeypatch):
    """ClassificationModel hard-codes pretrained=True; keep tests offline."""
    create_model = timm.create_model

    def offline_create_model(*args, **kwargs):
        return create_model(*args, **{**kwargs, "pretrained": False})

    monkeypatch.setattr(models.timm, "create_model", offline_create_model)


def _fit_one_batch(model, loader):
    trainer = pl.Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    return trainer


def _save_checkpoint(model, loader, path):
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    trainer.save_checkpoint(path)
    return str(path)


@pytest.fixture
def segmentation_loader():
    torch.manual_seed(0)
    images = torch.randn(4, 1, 32, 32)
    masks = (torch.rand(4, 1, 32, 32) > 0.5).float()
    return DataLoader(TensorDataset(images, masks), batch_size=2)


# --- segmentation -----------------------------------------------------------------


def test_binary_segmentation_model_forward_and_predict():
    model = dlt.create_segmentation_model(input_channels=2, **SEGMENTATION_KWARGS)
    model.eval()
    images = torch.randn(2, 2, 32, 32)
    probabilities = model(images)
    assert probabilities.shape == (2, 1, 32, 32)
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    prediction = model.predict_step(images)
    assert prediction.dtype == torch.bool
    assert isinstance(model.criterion, FocalTverskyLoss)


def test_multiclass_segmentation_model_adds_background_class():
    model = dlt.create_segmentation_model(n_classes=3, **SEGMENTATION_KWARGS).eval()
    images = torch.randn(1, 1, 32, 32)
    assert model(images).shape == (1, 4, 32, 32)
    assert model.predict_step(images).shape == (1, 32, 32)


def test_binary_segmentation_model_trains_one_step(segmentation_loader):
    model = dlt.create_segmentation_model(**SEGMENTATION_KWARGS)
    trainer = _fit_one_batch(model, segmentation_loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert "val_f1_score" in trainer.callback_metrics


def test_multiclass_segmentation_model_trains_one_step():
    torch.manual_seed(0)
    images = torch.randn(2, 1, 32, 32)
    masks = torch.randint(0, 3, (2, 1, 32, 32))
    loader = DataLoader(TensorDataset(images, masks), batch_size=2)
    model = models.SegmentationModel(
        input_channels=1,
        n_classes=2,
        learning_rate=1e-3,
        normalization={"type": "data_range"},
        ignore_index=-1,
        **SEGMENTATION_KWARGS,
    )
    trainer = _fit_one_batch(model, loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])


def test_segmentation_checkpoint_roundtrip(segmentation_loader, tmp_path):
    model = dlt.create_segmentation_model(
        normalization={"type": "data_range"}, **SEGMENTATION_KWARGS
    )
    path = _save_checkpoint(model, segmentation_loader, tmp_path / "seg.ckpt")
    loaded = dlt.load_segmentation_model_from_checkpoint(path)
    assert loaded.normalization == {"type": "data_range"}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_create_segmentation_model_from_checkpoint_overrides_hyperparameters(
    segmentation_loader, tmp_path
):
    model = dlt.create_segmentation_model(**SEGMENTATION_KWARGS)
    path = _save_checkpoint(model, segmentation_loader, tmp_path / "seg.ckpt")
    loaded = dlt.create_segmentation_model(
        checkpoint_path=path,
        learning_rate=0.5,
        normalization={"type": "mean_std", "mean": 0, "std": 1},
        **SEGMENTATION_KWARGS,
    )
    assert loaded.learning_rate == 0.5
    assert loaded.normalization["type"] == "mean_std"


@pytest.mark.parametrize(
    "overrides",
    [{"architecture": "FPN"}, {"encoder": "mobilenet_v2"}],
    ids=["architecture", "encoder"],
)
def test_create_segmentation_model_rejects_mismatched_checkpoint(
    segmentation_loader, tmp_path, overrides
):
    model = dlt.create_segmentation_model(**SEGMENTATION_KWARGS)
    path = _save_checkpoint(model, segmentation_loader, tmp_path / "seg.ckpt")
    with pytest.raises(ValueError, match="does not match"):
        dlt.create_segmentation_model(
            checkpoint_path=path, **{**SEGMENTATION_KWARGS, **overrides}
        )


def test_load_segmentation_model_from_bad_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="Could not load model"):
        dlt.load_segmentation_model_from_checkpoint(str(path))


# --- classification ------------------------------------------------------------------


@pytest.mark.parametrize(
    "classes, labels",
    [
        (["worm", "egg"], torch.tensor([[0.0], [1.0]])),
        (["worm", "egg", "error"], torch.tensor([0, 2])),
    ],
    ids=["binary", "multiclass"],
)
def test_classification_model_trains_and_predicts(classes, labels, tmp_path):
    n_outputs = 1 if len(classes) == 2 else len(classes)
    model = dlt.create_classification_model("resnet18", 1, classes)
    # the binary model is built with 2 outputs; replace its head to match BCE labels
    if len(classes) == 2:
        model.model.reset_classifier(n_outputs)
    loader = DataLoader(TensorDataset(torch.randn(2, 1, 32, 32), labels), batch_size=2)
    trainer = _fit_one_batch(model, loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    probabilities = model.eval()(torch.randn(3, 1, 32, 32))
    assert probabilities.shape == (3, n_outputs)


def test_classification_checkpoint_roundtrip(tmp_path):
    model = dlt.create_classification_model("resnet18", 1, ["a", "b", "c"])
    loader = DataLoader(
        TensorDataset(torch.randn(2, 1, 32, 32), torch.tensor([0, 1])), batch_size=2
    )
    path = _save_checkpoint(model, loader, tmp_path / "cls.ckpt")
    loaded = dlt.create_classification_model("ignored", 99, [], checkpoint_path=path)
    assert loaded.classes == ["a", "b", "c"]


# --- keypoint detection ----------------------------------------------------------------


@pytest.fixture
def keypoint_loader():
    rng = torch.Generator().manual_seed(0)
    inputs = [torch.randn(60, generator=rng).numpy() for _ in range(4)]
    heatmaps = [torch.rand(2, 60, generator=rng).numpy() for _ in range(4)]
    indices = [torch.tensor([10.0, 20.0]).numpy() for _ in range(4)]
    dataset = KeypointDetection1DTrainingDataset(inputs, heatmaps, indices)
    return DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "sigmoid", "none"])
def test_keypoint_model_forward_shapes(activation):
    model = dlt.create_keypoint_detection_model(1, 2, activation=activation).eval()
    heatmap, presence = model(torch.randn(3, 1, 64))
    assert heatmap.shape == (3, 2, 64)
    assert presence.shape == (3, 2)


def test_keypoint_model_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        dlt.create_keypoint_detection_model(1, 2, activation="tanh")


def test_keypoint_model_trains_and_predicts(keypoint_loader):
    model = dlt.create_keypoint_detection_model(1, 2)
    trainer = _fit_one_batch(model, keypoint_loader)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    series, valid_mask, *_ = next(iter(keypoint_loader))
    heatmap, presence = model.eval().predict_step((series, valid_mask))
    assert heatmap.shape == (2, 2, 64)
    assert ((presence >= 0) & (presence <= 1)).all()


def test_keypoint_checkpoint_roundtrip(keypoint_loader, tmp_path):
    model = dlt.create_keypoint_detection_model(1, 2, learning_rate=0.01)
    path = _save_checkpoint(model, keypoint_loader, tmp_path / "kp.ckpt")
    for loaded in (
        dlt.create_keypoint_detection_model(9, 9, checkpoint_path=path),
        dlt.load_keypoint_detection_model_from_checkpoint(path),
    ):
        assert loaded.learning_rate == 0.01


def test_load_keypoint_model_from_bad_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="Could not load keypoint"):
        dlt.load_keypoint_detection_model_from_checkpoint(str(path))

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score

from align_toolbox.deep_learning.utils.loss import (
    FocalTverskyLoss,
    MoltDetectionLoss,
    MultiClassFocalLoss,
)

from .archs import Unet1D


class ClassificationModel(pl.LightningModule):
    """
    PyTorch Lightning module for image classification using a pretrained backbone.

    Uses ``timm`` to load the specified architecture with ImageNet-pretrained weights.
    Applies ``BCEWithLogitsLoss`` + ``BinaryF1Score`` for binary tasks, or
    ``CrossEntropyLoss`` + ``MulticlassF1Score`` for multiclass tasks.

    Parameters:
        architecture (str): ``timm`` model name (e.g. ``"efficientnet_b0"``).
        input_channels (int): Number of input image channels.
        classes (list[str]): Class labels; ``len(classes)`` determines binary vs multiclass.
        learning_rate (float): Learning rate for the Adam optimizer.
        normalization (dict): Normalization config stored as a hyperparameter and
            used at inference time to reconstruct the preprocessing pipeline.
    """

    def __init__(
        self,
        architecture,
        input_channels,
        classes,
        learning_rate,
        normalization,
    ):
        super().__init__()
        n_classes = len(classes)
        if n_classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        model = timm.create_model(
            architecture,
            pretrained=True,
            num_classes=n_classes,
            in_chans=input_channels,
        )

        self.model = model
        self.learning_rate = learning_rate
        self.classes = classes
        self.n_classes = n_classes
        if n_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
            self.f1_score = BinaryF1Score()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.f1_score = MulticlassF1Score(num_classes=n_classes)

        self.normalization = normalization
        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return self.activation(y)

    def log_tb_images(self, viz_batch) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

    def training_step(self, batch):
        if batch is None:
            return None
        x, y = batch
        y_hat = self.model(x)
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)

        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        y_hat = self.activation(y_hat)

        if self.n_classes > 2:
            y_hat = torch.argmax(y_hat, dim=1)

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "train_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        if batch is None:
            return None
        x, y = batch
        y_hat = self.model(x)
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)

        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        y_hat = self.activation(y_hat)

        if self.n_classes > 2:
            y_hat = torch.argmax(y_hat, dim=1)

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "val_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for image segmentation using a pretrained encoder.

    Uses ``segmentation_models_pytorch`` to build an encoder–decoder model.
    For binary tasks (``n_classes == 1``): sigmoid activation +
    ``FocalTverskyLoss`` + ``BinaryF1Score``. For multiclass tasks:
    softmax activation + ``MultiClassFocalLoss`` + ``MulticlassF1Score``.

    Parameters:
        input_channels (int): Number of input image channels.
        n_classes (int): Number of foreground segmentation classes.
        learning_rate (float): Learning rate for the Adam optimizer.
        architecture (str): ``smp`` architecture name (e.g. ``"Unet"``).
        encoder (str): Encoder backbone name (e.g. ``"resnet34"``).
        pretrained_weights (str): Dataset the encoder was pretrained on
            (e.g. ``"imagenet"``).
        normalization (dict): Normalization config stored as a hyperparameter
            and used at inference time to reconstruct the preprocessing pipeline.
        criterion (nn.Module, optional): Loss function. If ``None``,
            ``FocalTverskyLoss`` is used for binary tasks and
            ``MultiClassFocalLoss`` for multiclass. (default: None)
        ignore_index (int, optional): Target value to ignore in the loss and
            F1 score. (default: None)
    """

    def __init__(
        self,
        input_channels,
        n_classes,
        learning_rate,
        architecture,
        encoder,
        pretrained_weights,
        normalization,
        criterion=None,
        ignore_index=None,
    ):
        super().__init__()
        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        model = smp.create_model(
            arch=architecture,
            encoder_name=encoder,
            encoder_weights=pretrained_weights,
            in_channels=input_channels,
            classes=n_classes + 1 if n_classes > 1 else n_classes,
            activation=None,
        )

        self.model = model
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index

        if criterion is None:
            if n_classes == 1:
                self.criterion = FocalTverskyLoss(ignore_index=self.ignore_index)
            else:
                self.criterion = MultiClassFocalLoss(
                    ignore_index=self.ignore_index,
                    alpha=torch.tensor([0.1] + [0.75] * n_classes),
                )
        else:
            self.criterion = criterion

        if n_classes == 1:
            self.f1_score = BinaryF1Score(ignore_index=self.criterion.ignore_index)
        else:
            self.f1_score = MulticlassF1Score(
                num_classes=n_classes + 1,
                ignore_index=self.criterion.ignore_index,
            )

        self.normalization = normalization
        self.save_hyperparameters(ignore=["criterion"])
        self.n_classes = n_classes

    def forward(self, x):
        y = self.model(x)
        return self.activation(y)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        y_hat = self.activation(y_hat)
        if self.n_classes > 1 and y.dim() == 4 and y.shape[1] == 1:
            y = y.squeeze(1)

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "train_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        y_hat = self.activation(y_hat)
        if self.n_classes > 1 and y.dim() == 4 and y.shape[1] == 1:
            y = y.squeeze(1)

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "val_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

    def predict_step(self, batch):
        x = batch

        pred = self.forward(x)

        # binarize predictions
        if self.n_classes == 1:
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class KeypointDetection1DModel(pl.LightningModule):
    """
    PyTorch Lightning module for 1D keypoint detection using a U-Net architecture.

    Operates on 1D sequences (e.g. worm volume time series) with a :class:`Unet1D`
    backbone that outputs both a per-class heatmap of shape ``(B, n_classes, T)``
    and a per-class presence logit of shape ``(B, n_classes)``. The activation is
    applied to both outputs. Training and validation batches are the 5-tuple
    ``(series, valid_mask, heatmap_target, index_target, presence_target)``
    produced by :class:`KeypointDetection1DTrainingDataset`, while prediction
    batches are the ``(series, valid_mask)`` pair. ``valid_mask`` keeps padded
    positions out of both the presence pooling and the loss.

    Parameters:
        input_channels (int): Number of input sequence channels.
        n_classes (int): Number of keypoint classes (output channels).
        learning_rate (float): Learning rate for the Adam optimizer.
        activation (str, optional): Output activation applied to the heatmap and
            presence outputs; one of ``"relu"``, ``"leaky_relu"``, ``"sigmoid"``,
            or ``"none"``. (default: ``"sigmoid"``)
        criterion (nn.Module, optional): Loss function, called as
            ``criterion(valid_mask, predicted_heatmap, predicted_presence,
            heatmap_target, presence_target)``. If ``None``, ``MoltDetectionLoss``
            is used. (default: None)

    Raises:
        ValueError: If ``activation`` is not supported.
    """

    def __init__(
        self,
        input_channels,
        n_classes,
        learning_rate,
        activation="sigmoid",
        criterion=None,
    ):
        super().__init__()

        model = Unet1D(num_classes=n_classes, input_channels=input_channels)
        self.model = model
        self.learning_rate = learning_rate

        # Set a default ignore_index if not present
        if not hasattr(self, "ignore_index"):
            self.ignore_index = -100

        if criterion is None:
            self.criterion = MoltDetectionLoss()

        else:
            self.criterion = criterion

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.save_hyperparameters()

    def forward(self, x, mask=None):
        predicted_heatmap, predicted_presence = self.model(x, mask=mask)
        return self.activation(predicted_heatmap), self.activation(predicted_presence)

    def training_step(self, batch):
        x, valid_mask, heatmap_target, index_target, presence_target = batch
        predicted_heatmap, predicted_presence = self.model(x, mask=valid_mask)
        predicted_heatmap, predicted_presence = self.activation(
            predicted_heatmap
        ), self.activation(predicted_presence)
        loss = self.criterion(
            valid_mask,
            predicted_heatmap,
            predicted_presence,
            heatmap_target,
            presence_target,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch):
        x, valid_mask, heatmap_target, index_target, presence_target = batch
        predicted_heatmap, predicted_presence = self.model(x, mask=valid_mask)
        predicted_heatmap, predicted_presence = self.activation(
            predicted_heatmap
        ), self.activation(predicted_presence)
        loss = self.criterion(
            valid_mask,
            predicted_heatmap,
            predicted_presence,
            heatmap_target,
            presence_target,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def predict_step(self, batch):
        x, valid_mask = batch
        predicted_heatmap, predicted_presence = self.model(x, mask=valid_mask)
        predicted_heatmap, predicted_presence = self.activation(
            predicted_heatmap
        ), self.activation(predicted_presence)
        return predicted_heatmap, predicted_presence

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

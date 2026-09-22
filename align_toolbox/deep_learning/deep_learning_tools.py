from typing import Any

from align_toolbox.deep_learning.architectures import (
    ClassificationModel,
    KeypointDetection1DModel,
    SegmentationModel,
)
from align_toolbox.deep_learning.utils.util import get_input_channels_from_checkpoint


def create_classification_model(
    architecture: str,
    input_channels: int,
    classes: list[str],
    learning_rate: float = 1e-4,
    checkpoint_path: str | None = None,
    normalization: dict = {"type": "percentile", "lo": 1, "hi": 99},
) -> ClassificationModel:
    """
    Create a classification model.

    Parameters:
        architecture (str): ``timm`` model name (e.g. ``"efficientnet_b0"``).
        input_channels (int): Number of input image channels.
        classes (list[str]): List of class label strings.
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-4)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint; if
            provided, the model is loaded from the checkpoint and all other
            arguments are ignored. (default: None)
        normalization (dict, optional): Normalization config stored as a
            hyperparameter for inference. (default: percentile 1–99)

    Returns:
        ClassificationModel: Constructed or loaded classification model.
    """

    if checkpoint_path is not None:
        model = ClassificationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
        return model

    model = ClassificationModel(
        architecture,
        input_channels,
        classes,
        learning_rate,
        normalization,
    )
    return model


def create_segmentation_model(
    input_channels: int = 1,
    n_classes: int = 1,
    architecture: str = "UnetPlusPlus",
    encoder: str = "efficientnet-b4",
    pretrained_weights: str = "image-micronet",
    normalization: dict = {"type": "percentile", "lo": 1, "hi": 99},
    learning_rate: float = 1e-5,
    checkpoint_path: str | None = None,
    reset_optimizer: bool = True,
    criterion: Any | None = None,
) -> SegmentationModel:
    """
    Create a segmentation model with a pretrained encoder.

    Parameters:
        input_channels (int, optional): Number of input image channels. (default: 1)
        n_classes (int, optional): Number of foreground segmentation classes.
            (default: 1)
        architecture (str, optional): ``smp`` architecture name. (default: ``"UnetPlusPlus"``)
        encoder (str, optional): Encoder backbone name. (default: ``"efficientnet-b4"``)
        pretrained_weights (str, optional): Dataset the encoder was pretrained on.
            (default: ``"image-micronet"``)
        normalization (dict, optional): Normalization config. (default: percentile 1–99)
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-5)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint. If
            provided, the model is loaded from the checkpoint, then
            ``learning_rate`` and ``normalization`` are updated. (default: None)
        reset_optimizer (bool, optional): If ``True``, discard the optimizer
            state from the checkpoint. (default: True)
        criterion (nn.Module, optional): Loss function. If ``None``, a default
            is chosen based on ``n_classes``. (default: None)

    Returns:
        SegmentationModel: Constructed or loaded segmentation model.

    Raises:
        ValueError: If the checkpoint architecture or encoder does not match
            the requested one.
    """
    model = SegmentationModel(
        input_channels=input_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
        architecture=architecture,
        encoder=encoder,
        pretrained_weights=pretrained_weights,
        normalization=normalization,
        criterion=criterion,
    )

    if checkpoint_path is not None:
        loaded_model = SegmentationModel.load_from_checkpoint(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        if reset_optimizer:
            loaded_model.optimizer = None
            loaded_model.lr_scheduler = None

            loaded_model._optimizer = None
            loaded_model._lr_scheduler = None

        # change the learning rate and normalization
        loaded_model.learning_rate = learning_rate
        loaded_model.normalization = normalization

        # check if the architecture matches
        if not isinstance(loaded_model.model, model.model.__class__):
            raise ValueError(
                f"Checkpoint architecture {loaded_model.model.__class__} does not match the requested architecture {architecture}"
            )
        # check if the encoder matches
        if loaded_model.model.encoder.__class__ != model.model.encoder.__class__:
            raise ValueError(
                f"Checkpoint encoder architecture {type(loaded_model.model.encoder).__name__} does not match the requested encoder architecture {type(model.model.encoder).__name__}"
            )

        loaded_model.configure_optimizers()

        return loaded_model

    return model


def create_keypoint_detection_model(
    input_channels: int,
    n_classes: int,
    learning_rate: float = 1e-4,
    checkpoint_path: str | None = None,
    criterion: Any | None = None,
    activation: str = "sigmoid",
) -> KeypointDetection1DModel:
    """
    Create a 1D keypoint detection model.

    The model predicts, for every input series, a per-class heatmap and a
    per-class presence probability, using a 1D U-Net backbone.

    Parameters:
        input_channels (int): Number of input sequence channels.
        n_classes (int): Number of keypoint classes (output channels).
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-4)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint; if
            provided, the model is loaded from the checkpoint and all other
            arguments are ignored. (default: None)
        criterion (nn.Module, optional): Loss function, called as
            ``criterion(valid_mask, predicted_heatmap, predicted_presence,
            heatmap_target, presence_target)``. If ``None``, ``MoltDetectionLoss``
            is used. (default: None)
        activation (str, optional): Output activation applied to the heatmap and
            presence outputs; one of ``"relu"``, ``"leaky_relu"``, ``"sigmoid"``,
            or ``"none"``. (default: ``"sigmoid"``)

    Returns:
        KeypointDetection1DModel: Constructed or loaded keypoint detection model.
    """

    if checkpoint_path is not None:
        model = KeypointDetection1DModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
        return model

    model = KeypointDetection1DModel(
        input_channels,
        n_classes,
        learning_rate,
        activation=activation,
        criterion=criterion,
    )
    return model


def load_segmentation_model_from_checkpoint(
    checkpoint_path: str,
) -> SegmentationModel:
    """
    Load a segmentation model from a checkpoint.

    First tries a direct load; if that fails (e.g. mismatched ``input_channels``
    in the checkpoint metadata), infers the channel count from the checkpoint
    weights and retries with ``pretrained_weights=None``.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.

    Returns:
        SegmentationModel: Loaded segmentation model.

    Raises:
        ValueError: If both loading attempts fail.
    """
    try:
        return SegmentationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    except Exception as e:
        try:
            return SegmentationModel.load_from_checkpoint(
                checkpoint_path,
                input_channels=get_input_channels_from_checkpoint(checkpoint_path),
                pretrained_weights=None,
                weights_only=False,
            )
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )


def load_keypoint_detection_model_from_checkpoint(
    checkpoint_path: str,
) -> KeypointDetection1DModel:
    """
    Load a 1D keypoint detection model from a checkpoint.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.

    Returns:
        KeypointDetection1DModel: Loaded keypoint detection model.

    Raises:
        ValueError: If the model cannot be loaded from the checkpoint.
    """
    try:
        return KeypointDetection1DModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    except Exception as e:
        raise ValueError(
            f"Could not load keypoint detection model from checkpoint {checkpoint_path}. Error: {e}"
        )

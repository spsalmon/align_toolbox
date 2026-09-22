import os
from collections.abc import Iterator
from math import ceil, floor

import torch
from torch.serialization import safe_globals

# import torch.nn as nn
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding


def divide_batch(batch: torch.Tensor, n: int) -> Iterator[torch.Tensor]:
    """
    Yield successive mini-batches of size ``n`` from ``batch``.

    Parameters:
        batch (Tensor): Input batch tensor with the batch dimension as axis 0.
        n (int): Mini-batch size.

    Yields:
        Tensor: Slice of ``batch`` along axis 0 of at most ``n`` samples.
    """
    for i in range(0, batch.shape[0], n):
        yield batch[i : i + n, ::]  # noqa: E203


def get_closest_upper_multiple(dim: int | float, multiple: int) -> int:
    """
    Round ``dim`` up to the nearest multiple of ``multiple``.

    Parameters:
        dim (int or float): Value to round up.
        multiple (int): The multiple to round to.

    Returns:
        int: Smallest multiple of ``multiple`` that is >= ``dim``.
    """
    return int(multiple * ceil(dim / multiple))


def get_closest_lower_multiple(dim: int | float, multiple: int) -> int:
    """
    Round ``dim`` down to the nearest multiple of ``multiple``.

    Parameters:
        dim (int or float): Value to round down.
        multiple (int): The multiple to round to.

    Returns:
        int: Largest multiple of ``multiple`` that is <= ``dim``.
    """
    return int(multiple * floor(dim / multiple))


def adjust_tensor_dimensions(
    source_tensor: torch.Tensor,
    target_tensor_shape: tuple,
) -> torch.Tensor:
    """
    Squeeze ``source_tensor`` then unsqueeze it to match ``target_tensor_shape``.

    Parameters:
        source_tensor (Tensor): Tensor to reshape.
        target_tensor_shape (tuple[int, ...]): Desired shape of the output tensor.

    Returns:
        Tensor: Reshaped tensor compatible with ``target_tensor_shape``.
    """
    # Squeeze out unnecessary dimensions from source tensor
    adjusted_tensor = source_tensor.squeeze()
    # Add necessary dimensions to match the target tensor shape
    for dim, size in enumerate(target_tensor_shape):
        if dim >= len(adjusted_tensor.shape):
            adjusted_tensor = adjusted_tensor.unsqueeze(dim)
    return adjusted_tensor


def rename_keys_and_adjust_dimensions(
    model: torch.nn.Module,
    pretrained_model: dict,
) -> dict:
    """
    Map pretrained weights into a model with differently named or shaped parameters.

    Pairs keys from ``model.state_dict()`` with keys from ``pretrained_model`` by
    position, adjusting tensor shapes via :func:`adjust_tensor_dimensions` when
    they differ.

    Parameters:
        model (nn.Module): Target model whose state dict keys define the mapping.
        pretrained_model (dict): Source state dict (e.g. from
            ``torch.load(...)["state_dict"]``).

    Returns:
        dict: New state dict compatible with ``model.load_state_dict()``.

    Raises:
        AssertionError: If the number of keys in ``model`` and ``pretrained_model``
            do not match.
    """
    assert len(model.state_dict().keys()) == len(
        pretrained_model.keys()
    ), f"The number of keys in the model ({len(model.state_dict().keys())}) and the pretrained model ({len(pretrained_model.keys())}) do not match."
    new_state_dict = {}
    for key, old_key in zip(model.state_dict().keys(), pretrained_model.keys()):
        target_tensor = model.state_dict()[key]
        source_tensor = pretrained_model[old_key]
        # Adjust the dimensions of the source tensor to match the target tensor
        if source_tensor.shape != target_tensor.shape:
            source_tensor = adjust_tensor_dimensions(source_tensor, target_tensor.shape)
        new_state_dict[key] = source_tensor
    return new_state_dict


def get_input_channels_from_checkpoint(checkpoint_path: str) -> int:
    """
    Infer the number of input channels from a PyTorch Lightning checkpoint.

    Searches the state dict for the first convolutional weight tensor and returns
    its ``in_channels`` dimension (``weight.shape[1]``).

    Parameters:
        checkpoint_path (str): Path to the ``.ckpt`` checkpoint file.

    Returns:
        int: Number of input channels, or 0 if no convolutional layer was found.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    input_channels = 0
    for key in state_dict.keys():
        if "conv" in key and "weight" in key:
            input_channels = state_dict[key].shape[1]
            break
    return input_channels


def create_lightweight_checkpoint(input_path: str, output_path: str) -> dict:
    """
    Load a PyTorch Lightning checkpoint and save a lightweight version.

    Keeps only ``state_dict``, ``hyper_parameters``, and a small set of optional
    metadata keys (``epoch``, ``global_step``, ``pytorch-lightning_version``),
    discarding optimizer state and other large tensors.

    Parameters:
        input_path (str): Path to the full ``.ckpt`` checkpoint file.
        output_path (str): Destination path for the lightweight checkpoint.

    Returns:
        dict: The lightweight checkpoint dictionary that was saved to
            ``output_path``.
    """
    print(f"Loading checkpoint from: {input_path}")

    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Standard loading failed, trying with safe_globals: {e}")
        with safe_globals([getattr]):
            checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")

    # Create lightweight checkpoint with only essential information
    lightweight_checkpoint = {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
    }

    optional_keys = ["epoch", "global_step", "pytorch-lightning_version"]
    for key in optional_keys:
        if key in checkpoint:
            lightweight_checkpoint[key] = checkpoint[key]

    torch.save(lightweight_checkpoint, output_path)

    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    savings = ((original_size - new_size) / original_size) * 100

    print(f"Lightweight file size: {new_size:.2f} MB")
    print(f"Space saved: {savings:.1f}%")

    removed_keys = set(checkpoint.keys()) - set(lightweight_checkpoint.keys())
    if removed_keys:
        print(f"Removed keys: {removed_keys}")

    return lightweight_checkpoint

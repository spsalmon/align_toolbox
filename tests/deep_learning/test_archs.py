import pytest
import torch

from align_toolbox.deep_learning.architectures.archs import (
    DilatedBottleneck1D,
    Unet1D,
    VGGBlock1D,
)


@pytest.fixture
def unet():
    torch.manual_seed(0)
    return Unet1D(num_classes=4, input_channels=2).eval()


def test_vgg_block_changes_channels_and_keeps_length():
    assert VGGBlock1D(3, 8, 5)(torch.randn(2, 3, 20)).shape == (2, 5, 20)


def test_dilated_bottleneck_keeps_shape():
    assert DilatedBottleneck1D(6)(torch.randn(2, 6, 16)).shape == (2, 6, 16)


@pytest.mark.parametrize("length", [64, 128])
def test_unet1d_output_shapes(unet, length):
    heatmap, presence = unet(torch.randn(3, 2, length))
    assert heatmap.shape == (3, 4, length)
    assert presence.shape == (3, 4)


def test_unet1d_full_mask_matches_no_mask(unet):
    series = torch.randn(2, 2, 64)
    with torch.no_grad():
        _, unmasked = unet(series)
        _, fully_masked = unet(series, mask=torch.ones(2, 64, dtype=torch.bool))
    torch.testing.assert_close(unmasked, fully_masked)


def test_unet1d_mask_restricts_presence_pooling(unet):
    series = torch.randn(1, 2, 64)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :32] = True
    with torch.no_grad():
        heatmap_a, presence_a = unet(series, mask=mask)
        heatmap_b, presence_b = unet(series)
    torch.testing.assert_close(heatmap_a, heatmap_b)
    assert not torch.allclose(presence_a, presence_b)

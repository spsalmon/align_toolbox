import pytest
import torch

from align_toolbox.deep_learning.utils import util


def test_divide_batch_yields_chunks_covering_the_batch():
    batch = torch.arange(10).reshape(10, 1)
    chunks = list(util.divide_batch(batch, 4))
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert torch.equal(torch.cat(chunks), batch)


@pytest.mark.parametrize(
    "dim, upper, lower", [(64, 64, 64), (65, 96, 64), (31.5, 32, 0), (0, 0, 0)]
)
def test_closest_multiples(dim, upper, lower):
    assert util.get_closest_upper_multiple(dim, 32) == upper
    assert util.get_closest_lower_multiple(dim, 32) == lower


def test_adjust_tensor_dimensions_restores_trailing_singleton_dims():
    source = torch.ones(1, 8, 1)
    adjusted = util.adjust_tensor_dimensions(source, (8, 1, 1))
    assert adjusted.shape == (8, 1, 1)


def test_rename_keys_and_adjust_dimensions_maps_weights_by_position():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Conv1d(2, 4, 1))
    pretrained = {
        "a.weight": torch.full((2, 3), 1.0),
        "a.bias": torch.full((2,), 2.0),
        "b.weight": torch.full((4, 2), 3.0),  # missing the kernel dimension
        "b.bias": torch.full((4,), 4.0),
    }
    state_dict = util.rename_keys_and_adjust_dimensions(model, pretrained)
    model.load_state_dict(state_dict)
    assert list(state_dict) == ["0.weight", "0.bias", "1.weight", "1.bias"]
    assert torch.all(model[1].weight == 3.0)


def test_rename_keys_and_adjust_dimensions_rejects_key_count_mismatch():
    with pytest.raises(AssertionError):
        util.rename_keys_and_adjust_dimensions(torch.nn.Linear(3, 2), {})


def test_get_input_channels_from_checkpoint(tmp_path):
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.bn.weight": torch.ones(8),
                "model.conv_stem.weight": torch.ones(8, 5, 3, 3),
            }
        },
        path,
    )
    assert util.get_input_channels_from_checkpoint(str(path)) == 5


def test_get_input_channels_from_checkpoint_without_conv_returns_zero(tmp_path):
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"fc.weight": torch.ones(2, 2)}}, path)
    assert util.get_input_channels_from_checkpoint(str(path)) == 0


def test_create_lightweight_checkpoint_keeps_only_essentials(tmp_path):
    full = {
        "state_dict": {"w": torch.ones(3)},
        "hyper_parameters": {"lr": 1e-3},
        "epoch": 4,
        "optimizer_states": [{"big": torch.ones(10_000)}],
        "callbacks": {},
    }
    input_path, output_path = tmp_path / "full.ckpt", tmp_path / "light.ckpt"
    torch.save(full, input_path)
    light = util.create_lightweight_checkpoint(str(input_path), str(output_path))
    assert set(light) == {"state_dict", "hyper_parameters", "epoch"}
    reloaded = torch.load(output_path, weights_only=False)
    assert set(reloaded) == set(light)
    assert output_path.stat().st_size < input_path.stat().st_size

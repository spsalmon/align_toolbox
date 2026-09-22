import pytest
import torch
import torch.nn.functional as F

from align_toolbox.deep_learning.utils import loss


def test_focal_tversky_is_near_zero_for_perfect_prediction():
    targets = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    inputs = torch.tensor([[-50.0, 50.0, 50.0, -50.0]])
    criterion = loss.FocalTverskyLoss(smooth=1)
    assert criterion(inputs, targets).item() == pytest.approx(0.0, abs=1e-6)


def test_focal_tversky_penalizes_wrong_predictions_more():
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    criterion = loss.FocalTverskyLoss(smooth=1)
    good = criterion(torch.tensor([-5.0, 5.0, 5.0, -5.0]), targets)
    bad = criterion(torch.tensor([5.0, -5.0, -5.0, 5.0]), targets)
    assert bad > good


def test_focal_tversky_ignores_ignore_index_pixels():
    criterion = loss.FocalTverskyLoss(ignore_index=-1, smooth=1)
    targets = torch.tensor([1.0, 0.0, -1.0])
    inputs_a = torch.tensor([5.0, -5.0, 5.0])
    inputs_b = torch.tensor([5.0, -5.0, -5.0])
    assert criterion(inputs_a, targets) == criterion(inputs_b, targets)


def test_focal_tversky_without_activation_uses_probabilities():
    criterion = loss.FocalTverskyLoss(activation=False, smooth=0)
    targets = torch.tensor([1.0, 0.0])
    assert criterion(torch.tensor([1.0, 0.0]), targets).item() == pytest.approx(0.0)


def test_multiclass_focal_loss_with_gamma_zero_equals_cross_entropy():
    logits = torch.randn(6, 3, generator=torch.Generator().manual_seed(0))
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    criterion = loss.MultiClassFocalLoss(gamma=0.0)
    torch.testing.assert_close(
        criterion(logits, targets), F.cross_entropy(logits, targets)
    )


def test_multiclass_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[4.0, 0.0], [0.5, 0.0]])
    targets = torch.tensor([0, 0])
    per_example = loss.MultiClassFocalLoss(gamma=2.0, reduction="none")(logits, targets)
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    ratio = per_example / cross_entropy
    assert ratio[0] < ratio[1]


def test_multiclass_focal_loss_flattens_spatial_inputs_and_ignores_index():
    logits = torch.zeros(1, 2, 2, 2)
    targets = torch.tensor([[[0, 1], [-1, -1]]])
    criterion = loss.MultiClassFocalLoss(gamma=0.0, reduction="sum", ignore_index=-1)
    assert criterion(logits, targets).item() == pytest.approx(
        2 * torch.log(torch.tensor(2.0))
    )


def test_multiclass_focal_loss_all_ignored_returns_zero():
    criterion = loss.MultiClassFocalLoss(ignore_index=-1)
    assert criterion(torch.zeros(2, 3), torch.tensor([-1, -1])).item() == 0.0


def test_multiclass_focal_loss_rejects_unknown_reduction():
    with pytest.raises(ValueError):
        loss.MultiClassFocalLoss(reduction="max")


def test_multiclass_focal_loss_repr_lists_arguments():
    assert "gamma=2.0" in repr(loss.MultiClassFocalLoss())


def test_bce_with_ignore_matches_bce_on_kept_elements():
    probabilities = torch.tensor([0.9, 0.2, 0.7, 0.4])
    targets = torch.tensor([1.0, 0.0, -1.0, 1.0])
    kept = torch.tensor([True, True, False, True])
    expected = F.binary_cross_entropy(probabilities[kept], targets[kept])
    torch.testing.assert_close(
        loss.BCELossWithIgnore()(probabilities, targets), expected
    )


def test_peak_weighted_mse_weights_errors_by_target():
    criterion = loss.PeakWeightedMSELoss(peak_weight=3.0)
    target = torch.tensor([0.0, 1.0])
    # errors of 1 at weight 1 and weight 4
    assert criterion(torch.tensor([1.0, 0.0]), target).item() == pytest.approx(2.5)


def test_molt_detection_loss_ignores_padded_frames():
    criterion = loss.MoltDetectionLoss()
    valid_mask = torch.tensor([[True, True, False, False]])
    target_heatmap = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    presence = torch.tensor([[0.9]])
    target_presence = torch.tensor([[1.0]])
    perfect = target_heatmap.clone()
    garbage_in_padding = perfect.clone()
    garbage_in_padding[..., 2:] = 1.0
    loss_perfect = criterion(
        valid_mask, perfect, presence, target_heatmap, target_presence
    )
    loss_padded = criterion(
        valid_mask, garbage_in_padding, presence, target_heatmap, target_presence
    )
    assert loss_perfect == loss_padded
    # only the presence term remains
    assert loss_perfect.item() == pytest.approx(-torch.log(torch.tensor(0.9)).item())


def test_molt_detection_loss_heatmap_term_counts_valid_frames():
    criterion = loss.MoltDetectionLoss(peak_weight=0.0)
    valid_mask = torch.tensor([[True, True, True, True]])
    predicted = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    target = torch.zeros(1, 1, 4)
    presence = torch.tensor([[1.0]])
    value = criterion(valid_mask, predicted, presence, target, torch.tensor([[1.0]]))
    assert value.item() == pytest.approx(2 / 4)

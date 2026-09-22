import numpy as np
import pytest

from align_toolbox.foundation import detect_molts


def test_interpolate_peaks_removes_single_frame_spikes():
    signal = np.linspace(0, 1, 30)
    spiky = signal.copy()
    spiky[10] += 5
    np.testing.assert_allclose(detect_molts.interpolate_peaks(spiky), signal)


def test_interpolate_peaks_does_not_mutate_input():
    signal = np.linspace(0, 1, 30)
    signal[10] += 5
    original = signal.copy()
    detect_molts.interpolate_peaks(signal)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize(
    "worm_types, expected",
    [
        (["egg", "egg", "worm", "worm"], 2),
        (["egg", "error", "egg", "error", "worm"], 3),
        (["worm", "worm"], np.nan),
        (["egg", "egg"], np.nan),
    ],
)
def test_find_hatch_time(worm_types, expected):
    result = detect_molts.find_hatch_time(np.array(worm_types))
    np.testing.assert_equal(result, expected)


def test_find_molts_detects_hatch_and_four_ecdyses(molting_volume, molt_centers):
    volume, worm_types = molting_volume
    molt_sizes = list(volume[list(molt_centers)])
    ecdysis = detect_molts.find_molts(volume, worm_types, molt_size_range=molt_sizes)
    assert list(ecdysis) == ["hatch_time", "M1", "M2", "M3", "M4"]
    assert ecdysis["hatch_time"] == 20
    for molt, center in zip(["M1", "M2", "M3", "M4"], molt_centers):
        # ecdysis happens at the end of the lethargus plateau
        assert center <= ecdysis[molt] <= center + 20


def test_find_molts_ignores_error_frames(molting_volume, molt_centers):
    volume, worm_types = molting_volume
    volume, worm_types = volume.copy(), worm_types.copy()
    volume[[200, 201, 300]] = 1.0
    worm_types[[200, 201, 300]] = "error"
    ecdysis = detect_molts.find_molts(
        volume, worm_types, molt_size_range=list(volume[list(molt_centers)])
    )
    for molt, center in zip(["M1", "M2", "M3", "M4"], molt_centers):
        assert center <= ecdysis[molt] <= center + 20


def test_find_molts_misses_molts_far_from_expected_sizes(molting_volume):
    volume, worm_types = molting_volume
    ecdysis = detect_molts.find_molts(
        volume, worm_types, molt_size_range=[1e9, 2e9, 3e9, 4e9]
    )
    assert all(np.isnan(ecdysis[m]) for m in ["M1", "M2", "M3", "M4"])


def test_find_end_molts_returns_nan_for_missing_mid_molts(molting_volume):
    volume, _ = molting_volume
    end_molts = detect_molts.find_end_molts(
        volume, np.array([120.0, np.nan, 360.0, np.nan])
    )
    assert np.isnan(end_molts[[1, 3]]).all()
    assert np.isfinite(end_molts[[0, 2]]).all()

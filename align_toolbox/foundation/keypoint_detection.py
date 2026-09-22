import numpy as np


def heatmap_to_keypoints_1D(
    heatmap: np.ndarray,
    presence: np.ndarray,
    height_threshold: float = 0.25,
    presence_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert a 1D heatmap to keypoints by finding the index of the highest value for each class, subject to a height threshold and presence threshold.
    Presence corresponds to the probability of the class being present in the signal.

    Parameters:
        heatmap (np.ndarray): 2D array of shape ``(n_classes, length)`` where each
            row is a 1D signal for one class.
        presence (np.ndarray): 1D array of shape ``(n_classes,)`` giving the
            probability that each class is present in the signal.
        height_threshold (float, optional): Minimum peak height to consider.
            (default: 0.25)
        presence_threshold (float, optional): Minimum presence probability below
            which the class is considered absent and its keypoint is set to NaN.
            (default: 0.5)

    Returns:
        np.ndarray: 1D array of shape ``(n_classes,)`` containing the index of the
            highest peak for each class, or NaN if the class was considered absent
            or no peak passed the height threshold.
    """
    peaks = []
    for i, (heatmap_i, presence_i) in enumerate(zip(heatmap, presence)):
        if presence_i < presence_threshold:
            peaks.append(np.nan)
            continue
        try:
            peak = np.argmax(heatmap_i)
            if heatmap_i[peak] < height_threshold:
                peaks.append(np.nan)
            else:
                peaks.append(peak)
        except ValueError:
            peaks.append(np.nan)
    peaks = np.array(peaks)
    return peaks

import importlib

import pytest

import align_toolbox


def test_version_is_resolved_from_metadata():
    assert align_toolbox.__version__ != "unknown"


@pytest.mark.parametrize(
    "module_name",
    [
        "align_toolbox.segmentation",
        "align_toolbox.straightening",
        "align_toolbox.quantification",
        "align_toolbox.classification",
        "align_toolbox.data_analysis",
        "align_toolbox.deep_learning",
        "align_toolbox.deep_learning.architectures",
    ],
)
def test_every_name_in_dunder_all_is_importable(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []

"""
Environment smoke tests — verify the migration from TF/Keras to PyTorch succeeded.
These require no data files and run in seconds.

Run with:  pytest tests/unit/test_environment.py -v
"""
import pytest


def test_tensorflow_not_installed():
    """TF was removed from the environment — importing it must fail."""
    with pytest.raises(ImportError):
        import tensorflow


def test_pytorch_importable():
    import torch
    major, minor, *_ = torch.__version__.split(".")
    assert (int(major), int(minor)) >= (2, 5), (
        f"PyTorch >= 2.5 required, got {torch.__version__}"
    )


def test_caiman_version():
    import caiman
    parts = caiman.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (1, 12), (
        f"CaImAn >= 1.12.0 required, got {caiman.__version__}"
    )


def test_cnmf_params_accepts_project_params():
    """CNMFParams must accept every key in CNMF_PARAMS without raising."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from caiman.source_extraction.cnmf import params as cnmf_params
    import params as project_params

    # fr is derived at runtime — exclude it from the dict test
    test_dict = {k: v for k, v in project_params.CNMF_PARAMS.items() if k != "fr"}
    cnmf_params.CNMFParams(params_dict=test_dict)


def test_cellpose_importable():
    from importlib.metadata import version
    v = version("cellpose")
    major, *_ = v.split(".")
    assert int(major) >= 3, f"Cellpose >= 3 required, got {v}"

from copy import deepcopy

import pytest
import torch
from torch import nn

from dense_unet_3d.cli import _load_model_from_checkpoint, _TinyStub
from dense_unet_3d.model.config import canonical_model_config


def test_synthetic_checkpoint_cannot_impersonate_named_graph(tmp_path):
    path = tmp_path / "synthetic.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save({"model_state_dict": model.state_dict()}, path)
    config = {"model": canonical_model_config("figure_skip_reconstruction_v1")}
    with pytest.raises(ValueError, match="Synthetic stub"):
        _load_model_from_checkpoint(str(path), torch.device("cpu"), config)
    assert isinstance(_load_model_from_checkpoint(str(path), torch.device("cpu")), _TinyStub)


def test_malformed_graph_metadata_is_not_bypassed_by_stub_keys(tmp_path):
    path = tmp_path / "malformed.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"name": "bogus"},
            "model_fingerprint": "bogus",
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported model"):
        _load_model_from_checkpoint(str(path), torch.device("cpu"))


def test_inference_restores_execution_and_rejects_preprocessing_mismatch(tmp_path):
    path = tmp_path / "reference.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    execution = {"precision": "fp32", "tf32": False, "deterministic": True}
    dataset = {"sampling": "native_slabs", "resize_dims": {"D": 12, "H": 224, "W": 224}}
    torch.save(
        {"model_state_dict": model.state_dict(), "execution": execution, "dataset_config": dataset},
        path,
    )
    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.backends.cudnn.allow_tf32 = True
        _load_model_from_checkpoint(str(path), torch.device("cpu"))
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.are_deterministic_algorithms_enabled()
        wrong = deepcopy(dataset)
        wrong["sampling"] = "whole_volume"
        with pytest.raises(ValueError, match="preprocessing mismatch"):
            _load_model_from_checkpoint(str(path), torch.device("cpu"), {"dataset": wrong})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior[0]
        torch.backends.cudnn.allow_tf32 = prior[1]
        torch.backends.cudnn.benchmark = prior[2]
        torch.backends.cudnn.deterministic = prior[3]
        torch.use_deterministic_algorithms(prior[4])

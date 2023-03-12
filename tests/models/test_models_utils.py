import os

import pytest
import torch.version

from tempor.models import utils


class TestEnableReproducibility:
    @pytest.mark.parametrize(
        "torch_use_deterministic_algorithms, torch_set_cudnn_deterministic, torch_disable_cudnn_benchmark, "
        "torch_handle_rnn_cuda_randomness",
        [
            (True, True, True, True),
            (False, False, False, False),
        ],
    )
    def test_base_case(
        self,
        torch_use_deterministic_algorithms,
        torch_set_cudnn_deterministic,
        torch_disable_cudnn_benchmark,
        torch_handle_rnn_cuda_randomness,
    ):
        utils.enable_reproducibility(
            random_seed=42,
            ignore_python_hash_seed_warning=True,
            torch_use_deterministic_algorithms=torch_use_deterministic_algorithms,
            torch_set_cudnn_deterministic=torch_set_cudnn_deterministic,
            torch_disable_cudnn_benchmark=torch_disable_cudnn_benchmark,
            torch_handle_rnn_cuda_randomness=torch_handle_rnn_cuda_randomness,
        )

    def test_raise_hashseed_warning(self, monkeypatch):
        env_vars = dict(os.environ)
        env_vars.pop("PYTHONHASHSEED", None)
        monkeypatch.setattr(os, "environ", env_vars)

        with pytest.warns(UserWarning, match=".*PYTHONHASHSEED.*"):
            utils.enable_reproducibility(
                random_seed=42,
                ignore_python_hash_seed_warning=False,
            )

    @pytest.mark.parametrize(
        "cuda_version, expected_CLB, expected_CWC",
        [
            (None, "NOT_SET", "NOT_SET"),
            ("9.2", "NOT_SET", "NOT_SET"),
            ("10.0", "NOT_SET", "NOT_SET"),
            ("10.1", "1", "NOT_SET"),
            ("10.2", "NOT_SET", ":4096:2"),
            ("11.3", "NOT_SET", ":4096:2"),
        ],
    )
    def test_handle_rnn_cuda_randomness(self, cuda_version, expected_CLB, expected_CWC, monkeypatch):
        env_vars = dict(os.environ)
        env_vars["CUDA_LAUNCH_BLOCKING"] = "NOT_SET"
        env_vars["CUBLAS_WORKSPACE_CONFIG"] = "NOT_SET"
        monkeypatch.setattr(os, "environ", env_vars)
        monkeypatch.setattr(torch.version, "cuda", cuda_version)

        utils.enable_reproducibility(
            random_seed=42,
            ignore_python_hash_seed_warning=True,
            torch_handle_rnn_cuda_randomness=True,
        )

        assert os.environ["CUDA_LAUNCH_BLOCKING"] == expected_CLB
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected_CWC

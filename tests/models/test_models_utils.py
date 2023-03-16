# pylint: disable=redefined-outer-name, unused-argument

import os

import pytest
import torch
import torch.backends.cudnn
import torch.version

from tempor.models import utils


@pytest.fixture(scope="function")
def reset_reproducibility(request):
    def teardown():
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    request.addfinalizer(teardown)


class TestEnableReproducibility:
    @pytest.mark.parametrize(
        "torch_use_deterministic_algorithms, torch_set_cudnn_deterministic, torch_disable_cudnn_benchmark",
        [
            (True, True, True),
            (False, False, False),
        ],
    )
    def test_base_case(
        self,
        torch_use_deterministic_algorithms,
        torch_set_cudnn_deterministic,
        torch_disable_cudnn_benchmark,
        reset_reproducibility,
    ):
        utils.enable_reproducibility(
            random_seed=42,
            torch_use_deterministic_algorithms=torch_use_deterministic_algorithms,
            torch_set_cudnn_deterministic=torch_set_cudnn_deterministic,
            torch_disable_cudnn_benchmark=torch_disable_cudnn_benchmark,
            warn_cuda_env_vars=False,
        )

    @pytest.mark.parametrize(
        "cuda_version, warn_CLB, warn_CWC",
        [
            (None, False, False),
            ("9.2", False, False),
            ("10.0", False, False),
            ("10.1", True, False),
            ("10.2", False, True),
            ("11.3", False, True),
        ],
    )
    def test_handle_rnn_cuda_randomness(
        self,
        cuda_version,
        warn_CLB,
        warn_CWC,
        monkeypatch,
        reset_reproducibility,
    ):
        env_vars = dict(os.environ)
        env_vars["CUDA_LAUNCH_BLOCKING"] = "NOT_SET"
        env_vars["CUBLAS_WORKSPACE_CONFIG"] = "NOT_SET"
        monkeypatch.setattr(os, "environ", env_vars)
        monkeypatch.setattr(torch.version, "cuda", cuda_version)

        if warn_CLB:
            with pytest.warns(UserWarning, match=".*CUDA_LAUNCH_BLOCKING.*"):
                utils.enable_reproducibility(
                    random_seed=42,
                    torch_use_deterministic_algorithms=True,
                )
        elif warn_CWC:
            with pytest.warns(UserWarning, match=".*CUBLAS_WORKSPACE_CONFIG.*"):
                utils.enable_reproducibility(
                    random_seed=42,
                    torch_use_deterministic_algorithms=True,
                )
        else:
            utils.enable_reproducibility(
                random_seed=42,
                torch_use_deterministic_algorithms=True,
            )

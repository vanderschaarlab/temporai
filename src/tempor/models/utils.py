import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.version


def enable_reproducibility(
    random_seed: int = 0,
    ignore_python_hash_seed_warning: bool = False,
    torch_use_deterministic_algorithms: bool = True,
    torch_set_cudnn_deterministic: bool = True,
    torch_disable_cudnn_benchmark: bool = True,
    torch_handle_rnn_cuda_randomness: bool = True,
) -> None:
    """Attempt to enable reproducibility of results by removing sources of non-determinism (randomness) wherever
    possible. This function does not guarantee reproducible results, as there could be many other sources of
    randomness, e.g. data splitting, third party libraries etc.

    The implementation is based on the information in PyTorch documentation here:
    https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        random_seed (int, optional):
            The random seed to set. Defaults to 0.
        ignore_python_hash_seed_warning (bool, optional):
            Unless this is set to True, will raise a ``UserWarning`` if ``"PYTHONHASHSEED"`` environment variable is
            not set. Defaults to False.
        torch_use_deterministic_algorithms (bool, optional):
            Whether to set ``torch.use_deterministic_algorithms(True)``. Defaults to True.
        torch_set_cudnn_deterministic (bool, optional):
            Whether to set ``torch.backends.cudnn.deterministic = True``. Defaults to True.
        torch_disable_cudnn_benchmark (bool, optional):
            Whether to set ``torch.backends.cudnn.benchmark = False``. Defaults to True.
        torch_handle_rnn_cuda_randomness (bool, optional):
            Handle the additional source of CUDA randomness in the RNN/LSTM implementations. See
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM. Note that this may overwrite
            the environment variables ``"CUDA_LAUNCH_BLOCKING"`` or ``"CUBLAS_WORKSPACE_CONFIG"``. Defaults to True.
    """
    # Python hash seed (only raise warning).
    if not ignore_python_hash_seed_warning:
        if not os.getenv("PYTHONHASHSEED"):
            warnings.warn(
                "PYTHONHASHSEED environmental variable has not been set in your environment, "
                "this could lead to randomness in certain situations",
                UserWarning,
            )

    # Built-in random module.
    random.seed(random_seed)

    # NumPy:
    np.random.seed(random_seed)

    # PyTorch:
    # Main seed:
    torch.manual_seed(random_seed)
    # Cuda seed, even if multiple GPUs:
    torch.cuda.manual_seed_all(random_seed)
    # If enabled, force deterministic algorithms:
    if torch_use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    # If enabled, set the CuDNN deterministic option.
    if torch_set_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    # If enabled, disable CuDNN benchmarking process to avoid possible non-determinism:
    if torch_disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    # Deal with RNN/LSTM-specific source of randomness on CUDA:
    if torch_handle_rnn_cuda_randomness:
        if torch.version.cuda not in ("", None):
            major, minor = [int(x) for x in torch.version.cuda.split(".")]
            if (major, minor) == (10, 1):
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            if major > 10 or (major == 10 and minor >= 2):
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

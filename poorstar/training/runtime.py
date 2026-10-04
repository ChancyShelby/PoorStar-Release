"""Training: runtime."""

import os
from collections import OrderedDict

import torch
from torch import nn


def runtime_profile(device):
    device = torch.device(device)
    return dict(
        schema="poorstar_c3_bc_runtime_v1",
        device=device.type,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if device.type == "cuda"
        else None,
        threads=torch.get_num_threads(),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )


def configure_bc_runtime(device, mode="DETERMINISTIC_V1"):
    if mode != "DETERMINISTIC_V1":
        raise ValueError("unknown BC runtime mode")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("qualified CPU or CUDA BC runtime required")
    workspace = ":4096:8"
    if device.type == "cuda":
        if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
            raise ValueError("configure BC cuBLAS workspace before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    return runtime_profile(device)


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Same adaptive rectangular averages, using deterministic matrix reductions.

    Each bin spans floor(i*L/B):ceil((i+1)*L/B), including overlapping bins
    for nondivisible sizes and upsampling. Separable weights preserve the exact
    real-valued function and gradient; floating reduction order can differ from
    ATen's atomic CUDA backward. No parameters, learned state or actor features
    are cached. Only bounded constant geometry matrices are reused.
    """

    def __init__(self, output_size):
        super().__init__()
        if (
            not isinstance(output_size, tuple)
            or len(output_size) != 2
            or any(type(n) is not int or n < 1 for n in output_size)
        ):
            raise ValueError("positive declared pool dimensions required")
        self.output_size = output_size
        self._geometry = OrderedDict()

    def _weights(self, length, bins, x):
        key = (length, bins, x.device, x.dtype)
        if key not in self._geometry:
            weights = x.new_zeros((bins, length))
            for i in range(bins):
                start = i * length // bins
                end = ((i + 1) * length + bins - 1) // bins
                weights[i, start:end] = 1.0 / (end - start)
            self._geometry[key] = weights
        self._geometry.move_to_end(key)
        while len(self._geometry) > 4:
            self._geometry.popitem(last=False)
        return self._geometry[key]

    def forward(self, x):
        if x.ndim not in (3, 4) or min(x.shape[-2:]) < 1 or not x.is_floating_point():
            raise ValueError("nonempty floating CHW or NCHW spatial input required")
        rows = self._weights(x.shape[-2], self.output_size[0], x)
        columns = self._weights(x.shape[-1], self.output_size[1], x)
        return torch.matmul(torch.matmul(rows, x), columns.transpose(0, 1))

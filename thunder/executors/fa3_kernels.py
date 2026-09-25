"""Opt-in Flash Attention 3 executor using prebuilt Hugging Face kernels."""

from functools import cache

import torch

import thunder
from thunder.core.devices import to_torch_device
from thunder.core.transforms import get_grad, put_grads
from thunder.extend import OperatorExecutor, register_executor


@cache
def _get_kernel():
    # Importing this executor must not download a kernel or require kernels.
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise ImportError("The fa3_kernels executor requires `pip install kernels`.") from exc

    return get_kernel("kernels-community/flash-attn3", version=1).flash_attn_interface


fa3_kernels_ex = OperatorExecutor("fa3_kernels", version="0.1")
register_executor(fa3_kernels_ex)


def _fwd_meta(q, k, v, causal=False, softmax_scale=None):
    return thunder.TensorProxy(like=q), thunder.TensorProxy(like=q, shape=q.shape[:-1], dtype=thunder.float32)


def _fwd_impl(q, k, v, causal=False, softmax_scale=None):
    # SDPA uses BHSD; Flash Attention uses BSHD.
    q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    out, lse, *_ = _get_kernel()._flash_attn_forward(q, k, v, softmax_scale=softmax_scale, causal=causal)
    return out.transpose(1, 2), lse


def _bwd_meta(dout, q, k, v, out, lse, causal=False, softmax_scale=None):
    return tuple(thunder.TensorProxy(like=x) for x in (q, k, v))


def _bwd_impl(dout, q, k, v, out, lse, causal=False, softmax_scale=None):
    dout, q, k, v, out = (x.transpose(1, 2).contiguous() for x in (dout, q, k, v, out))
    # The backward kernel writes into these buffers; never alias the inputs.
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    _get_kernel()._flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        lse.contiguous(),
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
    )
    return tuple(x.transpose(1, 2) for x in (dq, dk, dv))


_fwd = fa3_kernels_ex.register_operator("fa3_kernels_fwd", meta=_fwd_meta, fn=_fwd_impl)
_bwd = fa3_kernels_ex.register_operator("fa3_kernels_bwd", meta=_bwd_meta, fn=_bwd_impl)


def _checker(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if attn_mask is not None or dropout_p != 0.0:
        return False
    if any(x.ndim != 4 or any(d == 0 for d in x.shape) for x in (query, key, value)):
        return False
    if query.device.devicetype != thunder.devices.DeviceType.CUDA or any(
        x.device != query.device for x in (key, value)
    ):
        return False
    if query.dtype not in (thunder.float16, thunder.bfloat16) or any(x.dtype != query.dtype for x in (key, value)):
        return False
    # Keep the initial integration to equal heads and supported backward dimensions.
    if (
        query.shape[-1] not in (64, 128)
        or any(x.shape[:2] != query.shape[:2] or x.shape[-1] != query.shape[-1] for x in (key, value))
        or key.shape != value.shape
    ):
        return False
    # FA3 aligns a rectangular causal mask at the bottom right, unlike PyTorch.
    if is_causal and query.shape[-2] != key.shape[-2]:
        return False
    # Check the input device, not the process's current CUDA device.
    if torch.cuda.get_device_capability(to_torch_device(query.device)) != (9, 0):
        return False
    # Load before execution/CUDA graph capture. Explicit opt-in means loader errors
    # (missing dependency, incompatible build, offline cache miss) are actionable.
    _get_kernel()
    return True


def _execution_transform(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
    out, _ = _fwd(q, k, v, is_causal, scale)
    return out


def _grad_transform(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
    out, lse = _fwd(q, k, v, is_causal, scale)
    grads = _bwd(get_grad(out), q, k, v, out, lse, is_causal, scale)
    put_grads((q, k, v), grads)
    return out


fa3_kernels_ex.register_implementation(
    thunder.torch.scaled_dot_product_attention,
    checker=_checker,
    execution_transform=_execution_transform,
    grad_transform=_grad_transform,
)

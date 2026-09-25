import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import thunder
from thunder.executors import fa3_kernels as ex


def test_kernel_loader_is_optional_and_cached(monkeypatch):
    ex._get_kernel.cache_clear()
    try:
        monkeypatch.setitem(sys.modules, "kernels", None)
        with pytest.raises(ImportError, match="pip install kernels"):
            ex._get_kernel()
        interface = object()
        loader = Mock(return_value=SimpleNamespace(flash_attn_interface=interface))
        monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=loader))
        assert ex._get_kernel() is interface
        assert ex._get_kernel() is interface
        loader.assert_called_once_with("kernels-community/flash-attn3", version=1)
    finally:
        ex._get_kernel.cache_clear()


def test_kernel_loader_propagates_errors(monkeypatch):
    ex._get_kernel.cache_clear()
    loader = Mock(side_effect=RuntimeError("No compatible build"))
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=loader))
    with pytest.raises(RuntimeError, match="No compatible build"):
        ex._get_kernel()


def _proxy(shape=(2, 3, 5, 64), dtype=thunder.float16, device="cuda:1"):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=thunder.devices.Device(device))


@pytest.mark.parametrize(
    "change",
    [
        "cpu",
        "device",
        "dtype",
        "mixed_dtype",
        "rank",
        "empty",
        "head_dim",
        "heads",
        "batch",
        "kv",
        "mask",
        "dropout",
        "causal",
        "hardware",
    ],
)
def test_checker_rejects_without_loading(monkeypatch, change):
    q, k, v = (_proxy() for _ in range(3))
    kwargs = {}
    capability = Mock(return_value=(9, 0))
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(ex, "_get_kernel", loader)
    if change == "cpu":
        q, k, v = (_proxy(device="cpu") for _ in range(3))
    elif change == "device":
        k = _proxy(device="cuda:0")
    elif change == "dtype":
        q, k, v = (_proxy(dtype=thunder.float32) for _ in range(3))
    elif change == "mixed_dtype":
        k = _proxy(dtype=thunder.bfloat16)
    elif change == "rank":
        q = _proxy(shape=(3, 5, 64))
    elif change == "empty":
        q = _proxy(shape=(2, 3, 0, 64))
    elif change == "head_dim":
        q, k, v = (_proxy(shape=(2, 3, 5, 32)) for _ in range(3))
    elif change == "heads":
        k = v = _proxy(shape=(2, 1, 5, 64))
    elif change == "batch":
        k = v = _proxy(shape=(1, 3, 5, 64))
    elif change == "kv":
        v = _proxy(shape=(2, 3, 6, 64))
    elif change == "mask":
        kwargs["attn_mask"] = object()
    elif change == "dropout":
        kwargs["dropout_p"] = 0.1
    elif change == "causal":
        k = v = _proxy(shape=(2, 3, 7, 64))
        kwargs["is_causal"] = True
    elif change == "hardware":
        capability.return_value = (8, 0)
    assert not ex._checker(q, k, v, **kwargs)
    loader.assert_not_called()


def test_checker_uses_input_device(monkeypatch):
    capability = Mock(return_value=(9, 0))
    loader = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(ex, "_get_kernel", loader)
    assert ex._checker(_proxy(), _proxy(shape=(2, 3, 7, 64)), _proxy(shape=(2, 3, 7, 64)))
    capability.assert_called_once_with(torch.device("cuda:1"))
    loader.assert_called_once()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_adapter_layout_scale_and_gradient_buffers(monkeypatch, causal, scale):
    # Run the adapter with CPU tensors and a strict mock of the FA3 v1 interface.
    # Unequal head/sequence sizes detect an accidental BHSD/BSHD mixup.
    q, k, v = (torch.randn(2, 3, 5, 64) for _ in range(3))
    saved = tuple(x.clone() for x in (q, k, v))
    expected_scale = 64**-0.5 if scale is None else scale
    lse = torch.randn(2, 3, 5)

    def forward(q_, k_, v_, *, softmax_scale, causal):
        assert softmax_scale == expected_scale
        assert causal == test_causal
        for actual, original in zip((q_, k_, v_), saved):
            torch.testing.assert_close(actual, original.transpose(1, 2))
            assert actual.is_contiguous()
        return q_ + k_ + v_, lse, None, None

    def backward(dout, q_, k_, v_, out, lse_, *, dq, dk, dv, softmax_scale, is_causal):
        assert softmax_scale == expected_scale
        assert is_causal == test_causal
        torch.testing.assert_close(lse_, lse)
        torch.testing.assert_close(out, q_ + k_ + v_)
        for grad, original, factor in zip((dq, dk, dv), (q_, k_, v_), (1, 2, 3)):
            assert grad.data_ptr() != original.data_ptr()
            grad.copy_(dout * factor)

    test_causal = causal
    monkeypatch.setattr(
        ex, "_get_kernel", lambda: SimpleNamespace(_flash_attn_forward=forward, _flash_attn_backward=backward)
    )
    out, actual_lse = ex._fwd_impl(q, k, v, causal, scale)
    torch.testing.assert_close(out, q + k + v)
    assert actual_lse is lse
    dout = torch.randn_like(out)
    grads = ex._bwd_impl(dout, q, k, v, out, actual_lse, causal, scale)
    for grad, factor in zip(grads, (1, 2, 3)):
        torch.testing.assert_close(grad, dout * factor)
    for actual, original in zip((q, k, v), saved):
        torch.testing.assert_close(actual, original)


def test_compiled_forward_backward_with_mock_kernel(monkeypatch):
    # Exercise real Thunder tracing/autograd without requiring a downloaded GPU binary.
    impl = ex.fa3_kernels_ex.implmap[thunder.torch.scaled_dot_product_attention.id]
    monkeypatch.setattr(impl, "checker", lambda *args, **kwargs: True)

    def forward(q, k, v, *, softmax_scale, causal):
        return q + k + v, torch.zeros(q.shape[0], q.shape[2], q.shape[1]), None, None

    def backward(dout, q, k, v, out, lse, *, dq, dk, dv, softmax_scale, is_causal):
        for grad in (dq, dk, dv):
            grad.copy_(dout)

    monkeypatch.setattr(
        ex, "_get_kernel", lambda: SimpleNamespace(_flash_attn_forward=forward, _flash_attn_backward=backward)
    )
    fn = thunder.jit(torch.nn.functional.scaled_dot_product_attention, executors=[ex.fa3_kernels_ex])
    q, k, v = (torch.randn(2, 3, 5, 64, requires_grad=True) for _ in range(3))
    out = fn(q, k, v)
    torch.testing.assert_close(out, q + k + v)
    dout = torch.randn_like(out)
    for grad in torch.autograd.grad(out, (q, k, v), dout):
        torch.testing.assert_close(grad, dout)
    trace = thunder.last_traces(fn)[-1]
    fwd = next(b for b in trace.bound_symbols if b.sym.name == "fa3_kernels_fwd")
    assert fwd.output[1].shape == (2, 3, 5)
    assert fwd.output[1].dtype == thunder.float32
    assert any(b.sym.name == "fa3_kernels_bwd" for b in thunder.last_backward_traces(fn)[-1].bound_symbols)


def test_cpu_falls_back_without_kernel(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(ex, "_get_kernel", loader)
    fn = thunder.jit(torch.nn.functional.scaled_dot_product_attention, executors=[ex.fa3_kernels_ex])
    q, k, v = (torch.randn(2, 3, 5, 64) for _ in range(3))
    torch.testing.assert_close(fn(q, k, v), torch.nn.functional.scaled_dot_product_attention(q, k, v))
    loader.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal,kv_length", [(False, 11), (True, 7)])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_prebuilt_forward_backward(dtype, head_dim, causal, kv_length, scale):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")
    pytest.importorskip("kernels")
    # Sliced last dimensions also exercise the adapter's contiguous conversion.
    q = torch.randn(2, 3, 7, head_dim * 2, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    k, v = (torch.randn(2, 3, kv_length, head_dim, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2))
    saved = tuple(x.detach().clone() for x in (q, k, v))
    fn = thunder.jit(torch.nn.functional.scaled_dot_product_attention, executors=[ex.fa3_kernels_ex])
    result = fn(q, k, v, is_causal=causal, scale=scale)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, scale=scale
    )
    torch.testing.assert_close(result.float(), reference, atol=2e-2, rtol=2e-2)
    dout = torch.randn_like(result)
    grads = torch.autograd.grad(result, (q, k, v), dout)
    expected = torch.autograd.grad(reference, (q, k, v), dout.float())
    for actual, ref in zip(grads, expected):
        torch.testing.assert_close(actual, ref, atol=3e-2, rtol=3e-2)
    for actual, original in zip((q, k, v), saved):
        torch.testing.assert_close(actual, original)
    assert any(b.sym.name == "fa3_kernels_fwd" for b in thunder.last_traces(fn)[-1].bound_symbols)
    assert any(b.sym.name == "fa3_kernels_bwd" for b in thunder.last_backward_traces(fn)[-1].bound_symbols)

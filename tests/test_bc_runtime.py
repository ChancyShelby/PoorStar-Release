from dataclasses import replace

import pytest
import torch

from poorstar.training.runtime import (
    DeterministicAdaptiveAvgPool2d,
    configure_bc_runtime,
    runtime_profile,
)


@pytest.mark.parametrize("shape", [(2, 3, 16, 24), (2, 3, 11, 13), (3, 3, 5), (1, 2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pool_matches_native_rectangles_and_overlapping_gradients(shape, dtype):
    torch.manual_seed(43)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    pool = DeterministicAdaptiveAvgPool2d((8, 8))
    actual = pool(x)
    expected = torch.nn.functional.adaptive_avg_pool2d(y, (8, 8))
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    assert torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert torch.allclose(x.grad, y.grad, rtol=tolerance, atol=tolerance)
    assert not pool.state_dict() and not list(pool.parameters())
    assert all(not w.requires_grad for w in pool._geometry.values())


def test_geometry_cache_is_bounded_and_does_not_cache_input_activations():
    pool = DeterministicAdaptiveAvgPool2d((8, 8))
    for n in range(3, 15):
        x = torch.randn(1, 2, n, n + 1, requires_grad=True)
        pool(x).sum().backward()
        assert len(pool._geometry) <= 4
    x = torch.randn(1, 2, 11, 13)
    assert torch.equal(pool(x), pool(x.clone()))
    assert torch.equal(pool(torch.zeros_like(x)), torch.zeros(1, 2, 8, 8))


def test_equivalent_pool_preserves_model_parameters_objective_and_gradient():
    from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
    from poorstar.training.foundation import bc_objective
    from tests.test_coarse_bc import decision_rows

    rows, profile, catalog, _ = decision_rows()
    cfg = BCModelConfig.from_catalog(catalog, profile, hidden_size=32, maximum_delay=8)
    torch.manual_seed(43)
    model = RecurrentBCPolicy(cfg)
    native = RecurrentBCPolicy(replace(cfg, deterministic_spatial_pool=False))
    native.load_state_dict(model.state_dict())
    a = bc_objective(model, [rows], profile, collect_metrics=False, feature_batch_size=4)
    b = bc_objective(native, [rows], profile, collect_metrics=False, feature_batch_size=4)
    assert (
        a.denominators == b.denominators and a.counts == b.counts and a.rejections == b.rejections
    )
    assert torch.allclose(a.loss, b.loss, rtol=2e-5, atol=2e-6)
    a.loss.backward()
    b.loss.backward()
    for name, p in model.named_parameters():
        q = dict(native.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, rtol=3e-4, atol=3e-6)
    assert cfg.sha256 != native.config.sha256


def test_checkpoint_rejects_changed_runtime_before_restoring_weights(tmp_path, monkeypatch):
    from poorstar.data.bc.windows import WindowOrder, load_checkpoint, save_checkpoint
    from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
    from poorstar.training.foundation import BCTrainer
    from tests.test_coarse_bc import decision_rows

    rows, profile, catalog, _ = decision_rows()
    model = RecurrentBCPolicy(
        BCModelConfig.from_catalog(catalog, profile, hidden_size=32, maximum_delay=8)
    )
    trainer = BCTrainer(model, profile, learning_rate=0.001)
    order = WindowOrder(1, 43)
    trainer.step([rows])
    save_checkpoint(tmp_path / "one", trainer, order, {})
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actual = runtime_profile(model.device)
    monkeypatch.setattr(
        "poorstar.training.runtime.runtime_profile",
        lambda device: dict(actual, threads=actual["threads"] + 1),
    )
    with pytest.raises(ValueError, match="runtime drift"):
        load_checkpoint(tmp_path / "one", trainer, order, {})
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_runtime_mode_is_explicit_and_no_warn_only_escape():
    with pytest.raises(ValueError, match="runtime mode"):
        configure_bc_runtime("cpu", "RELAXED")
    with pytest.raises(ValueError, match="CPU or CUDA"):
        configure_bc_runtime("mps")
    from poorstar.data.bc.windows import BCRunPlan

    with pytest.raises(ValueError, match="runtime mode"):
        BCRunPlan("ENGINEERING", 43, 2, runtime_mode="RELAXED")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime qualification is separate")
def test_strict_cuda_pool_backward_repeats_exactly(monkeypatch):
    # The complete cloud runner sets this before importing/initializing CUDA.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old = runtime_profile("cuda")
    try:
        configured = configure_bc_runtime("cuda")
        assert configured["deterministic_algorithms"] and not configured["deterministic_warn_only"]
        x = (
            torch.arange(2 * 3 * 11 * 13, device="cuda", dtype=torch.float32).reshape(2, 3, 11, 13)
            / 100
        )
        pool = DeterministicAdaptiveAvgPool2d((8, 8))
        results = []
        for _ in range(2):
            a = x.detach().clone().requires_grad_()
            v = pool(a)
            v.square().sum().backward()
            results.append((v.detach().clone(), a.grad.clone()))
        assert all(torch.equal(a, b) for a, b in zip(*results))
    finally:
        torch.use_deterministic_algorithms(
            old["deterministic_algorithms"], warn_only=old["deterministic_warn_only"]
        )
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.set_num_threads(old["threads"])

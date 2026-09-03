from __future__ import annotations

from types import SimpleNamespace

import torch
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter


def _bare_adapter() -> ColBERTModernBERTFlashAdapter:
    """Construct without load() (which requires CUDA); set only what methods need."""
    return object.__new__(ColBERTModernBERTFlashAdapter)


def _fake_model(hidden_size: int, **config_extra) -> SimpleNamespace:
    """Minimal stand-in for the loaded backbone; only config attrs are read."""
    return SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size, **config_extra))


def test_project_applies_chain_then_truncates() -> None:
    """WITH a Dense chain ending at token_dim: sequential matmuls, then a no-op truncate."""
    adapter = _bare_adapter()
    adapter._token_dim = 4
    w0 = torch.randn(6, 8)  # [out_features, in_features]
    w1 = torch.randn(4, 6)
    adapter._dense_chain = [w0, w1]

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    expected = (hidden @ w0.T) @ w1.T
    assert torch.equal(out, expected)


def test_project_single_weight_equivalent_to_old_math() -> None:
    """A length-1 chain must be byte-identical to the pre-#1680 single-head math
    (``hidden @ W.T`` then truncate) — GTE/Reason representations must NOT change.
    """
    adapter = _bare_adapter()
    adapter._token_dim = 4
    weight = torch.randn(6, 8)
    adapter._dense_chain = [weight]

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    assert torch.equal(out, (hidden @ weight.T)[:, :4])


def test_project_pure_truncation_without_chain() -> None:
    """WITHOUT a chain: falls back to backbone truncation (backward-compatible)."""
    adapter = _bare_adapter()
    adapter._token_dim = 4
    adapter._dense_chain = None

    hidden = torch.randn(5, 8)
    out = adapter._project(hidden)

    assert out.shape == (5, 4)
    assert torch.equal(out, hidden[:, :4])


def test_compute_rope_uses_layer_appropriate_theta() -> None:
    """Global layers use global_rope_theta (160k), local layers local_rope_theta (10k).

    With different bases the rotation angles at position >= 1 must differ; at
    position 0 both are identity (cos=1, sin=0).
    """
    adapter = _bare_adapter()
    adapter._device = "cpu"
    adapter._compute_precision = "float32"
    adapter._model = _fake_model(
        hidden_size=8,
        num_attention_heads=2,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
    )

    position_ids = torch.tensor([0, 1, 2, 3])
    global_cos, global_sin = adapter._compute_rope(position_ids, use_global=True)
    local_cos, local_sin = adapter._compute_rope(position_ids, use_global=False)

    assert global_cos.shape == local_cos.shape == (4, 4)  # [total_tokens, head_dim]
    # Position 0 is base-independent (angle 0).
    assert torch.equal(global_cos[0], local_cos[0])
    assert torch.equal(global_sin[0], local_sin[0])
    # Positions >= 1 rotate at different frequencies for different bases.
    assert not torch.allclose(global_cos[1:], local_cos[1:])
    assert not torch.allclose(global_sin[1:], local_sin[1:])


_POSITIONS = torch.tensor([0, 1, 2, 3])


def _reference_cos_sin(theta: float, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of the adapter's rope math for a known theta."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(_POSITIONS.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rope_adapter(**config_extra) -> ColBERTModernBERTFlashAdapter:
    """CPU/fp32 adapter stub with flat rope attrs plus any nested extras."""
    adapter = _bare_adapter()
    adapter._device = "cpu"
    adapter._compute_precision = "float32"
    adapter._model = _fake_model(
        hidden_size=8,
        num_attention_heads=2,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        **config_extra,
    )
    return adapter


def test_compute_rope_prefers_nested_rope_parameters() -> None:
    """A transformers>=5 nested ``rope_parameters`` mapping wins over the flat
    attrs for BOTH layer kinds (5.x-serialized configs are authoritative about
    themselves; 4.x ModernBertConfig fills the flat attrs from class defaults).
    """
    adapter = _rope_adapter(
        rope_parameters={
            "full_attention": {"rope_theta": 111000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 222000.0, "rope_type": "default"},
        },
    )

    global_cos, global_sin = adapter._compute_rope(_POSITIONS, use_global=True)
    local_cos, local_sin = adapter._compute_rope(_POSITIONS, use_global=False)

    expected_global_cos, expected_global_sin = _reference_cos_sin(111000.0, head_dim=4)
    expected_local_cos, expected_local_sin = _reference_cos_sin(222000.0, head_dim=4)
    assert torch.equal(global_cos, expected_global_cos)
    assert torch.equal(global_sin, expected_global_sin)
    assert torch.equal(local_cos, expected_local_cos)
    assert torch.equal(local_sin, expected_local_sin)


def test_compute_rope_flat_attrs_when_nested_absent() -> None:
    """No ``rope_parameters`` key (every 4.x-written config): the flat-attr
    path stays byte-identical — regression pin for GTE/Reason/mxbai-edge.
    """
    adapter = _rope_adapter()

    global_cos, _ = adapter._compute_rope(_POSITIONS, use_global=True)
    local_cos, _ = adapter._compute_rope(_POSITIONS, use_global=False)

    expected_global_cos, _ = _reference_cos_sin(160000.0, head_dim=4)
    expected_local_cos, _ = _reference_cos_sin(10000.0, head_dim=4)
    assert torch.equal(global_cos, expected_global_cos)
    assert torch.equal(local_cos, expected_local_cos)


def test_compute_rope_malformed_nested_falls_back_to_flat() -> None:
    """Malformed nested declarations degrade to the flat attrs, never crash:
    non-mapping rope_parameters, non-mapping layer entry, non-numeric theta,
    and bool theta (bool is an int subclass but not a rope base).
    """
    malformed_variants = [
        {"rope_parameters": "sans_pos"},
        {"rope_parameters": {"full_attention": None, "sliding_attention": 42}},
        {"rope_parameters": {"full_attention": {"rope_theta": "fast"}, "sliding_attention": {}}},
        {"rope_parameters": {"full_attention": {"rope_theta": True}, "sliding_attention": {"rope_theta": True}}},
    ]
    expected_global_cos, _ = _reference_cos_sin(160000.0, head_dim=4)
    expected_local_cos, _ = _reference_cos_sin(10000.0, head_dim=4)

    for config_extra in malformed_variants:
        adapter = _rope_adapter(**config_extra)
        global_cos, _ = adapter._compute_rope(_POSITIONS, use_global=True)
        local_cos, _ = adapter._compute_rope(_POSITIONS, use_global=False)
        assert torch.equal(global_cos, expected_global_cos), config_extra
        assert torch.equal(local_cos, expected_local_cos), config_extra


def test_compute_rope_partial_nested_mixes_sources() -> None:
    """A nested mapping declaring only one layer kind covers that kind; the
    other keeps the flat attr.
    """
    adapter = _rope_adapter(
        rope_parameters={"sliding_attention": {"rope_theta": 222000.0}},
    )

    global_cos, _ = adapter._compute_rope(_POSITIONS, use_global=True)
    local_cos, _ = adapter._compute_rope(_POSITIONS, use_global=False)

    expected_global_cos, _ = _reference_cos_sin(160000.0, head_dim=4)  # flat
    expected_local_cos, _ = _reference_cos_sin(222000.0, head_dim=4)  # nested
    assert torch.equal(global_cos, expected_global_cos)
    assert torch.equal(local_cos, expected_local_cos)

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.deps import collect_bundle_deps
from typer.testing import CliRunner


def test_serve_passes_sie_pool_to_app_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setenv("SIE_POOL", "customer-a")
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    try:
        cli.serve(
            port=8080,
            host="127.0.0.1",
            device="cpu",
            models_dir=str(tmp_path),
            bundle=None,
            models=None,
            local_cache=None,
            cluster_cache=None,
            hf_fallback=True,
            reload=False,
            tracing=False,
            instrumentation=False,
            verbose=False,
            log_level="info",
            preload=None,
            json_logs=False,
        )
    finally:
        monkeypatch.delenv("SIE_BUNDLE", raising=False)
        monkeypatch.delenv("SIE_HF_FALLBACK", raising=False)
        monkeypatch.delenv("SIE_INSTRUMENTATION", raising=False)

    assert captured["config"].pool_name == "customer-a"


def _write_bundle(
    path: Path,
    *,
    name: str = "shared",
    adapters: list[str] | None = None,
    deps: dict[str, str] | None = None,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "adapters": adapters or ["sie_server.adapters.fake.adapter"],
                "deps": deps or {},
            }
        )
    )


def _write_model(
    path: Path,
    *,
    sie_id: str,
    pool: str | None = None,
    adapter: str = "sie_server.adapters.fake.adapter:FakeAdapter",
) -> None:
    data: dict[str, Any] = {
        "sie_id": sie_id,
        "package_backed": True,
        "tasks": {"extract": {}},
        "profiles": {
            "default": {
                "max_batch_tokens": 1024,
                "adapter_path": adapter,
            }
        },
    }
    if pool is not None:
        data["pool"] = pool
    path.write_text(yaml.safe_dump(data))


def _clear_unrelated_serve_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI tests independent from suite-level env round-trip tests."""
    monkeypatch.delenv("SIE_BUNDLE", raising=False)
    monkeypatch.delenv("SIE_DEVICE", raising=False)
    monkeypatch.delenv("SIE_EXTRA_MODELS", raising=False)
    monkeypatch.delenv("SIE_DEVICES", raising=False)
    monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
    monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
    monkeypatch.delenv("SIE_POOL", raising=False)
    monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)


def test_serve_preserves_env_backed_runtime_config_through_uvicorn_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Environment-backed serve config survives run_server's env round trip."""
    _clear_unrelated_serve_env(monkeypatch)
    model_ids = ["org/preload-a", "org/preload-b", "org/pinned-a", "org/pinned-b"]
    for index, model_id in enumerate(model_ids):
        _write_model(tmp_path / f"model-{index}.yaml", sie_id=model_id)
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        config.save_to_env_vars()
        captured["config"] = AppStateConfig.from_env_vars()

    monkeypatch.setenv("SIE_POOL", "customer-a")
    monkeypatch.setenv("SIE_DEVICES", "cpu")
    monkeypatch.setenv("SIE_PRELOAD_MODELS", "org/preload-a, org/preload-b")
    monkeypatch.setenv("SIE_PINNED_MODELS", "ORG/PINNED-A:default, org/pinned-b")
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    try:
        cli.serve(
            port=8080,
            host="127.0.0.1",
            device="cpu",
            models_dir=str(tmp_path),
            bundle=None,
            models=",".join(model_ids),
            local_cache=None,
            cluster_cache=None,
            hf_fallback=True,
            reload=False,
            tracing=False,
            instrumentation=False,
            verbose=False,
            log_level="info",
            preload=None,
            json_logs=False,
        )
    finally:
        monkeypatch.delenv("SIE_BUNDLE", raising=False)
        monkeypatch.delenv("SIE_HF_FALLBACK", raising=False)
        monkeypatch.delenv("SIE_INSTRUMENTATION", raising=False)

    config = captured["config"]
    assert config.devices == ["cpu"]
    assert config.pool_name == "customer-a"
    assert config.preload_models == ["org/preload-a", "org/preload-b"]
    assert config.pinned_models == ["ORG/PINNED-A:default", "org/pinned-b"]
    assert "Pinned (from env): 2 models will be kept resident" in capsys.readouterr().out


def test_serve_rejects_pinned_models_outside_model_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "fake.yaml", name="fake")
    _write_model(models_dir / "selected.yaml", sie_id="org/selected")
    called = False

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("SIE_PINNED_MODELS", "org/not-selected")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir), "--models", "org/selected"],
    )

    assert result.exit_code == 1
    assert "Pinned model(s) not in model filter: org/not-selected" in result.output
    assert not called


def test_load_bundle_filters_models_by_sie_pool(tmp_path: Path) -> None:
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "shared.yaml")
    _write_model(models_dir / "default.yaml", sie_id="org/default")
    _write_model(models_dir / "tenant.yaml", sie_id="org/tenant", pool="customer-a")

    assert cli.load_bundle("shared", bundles_dir, str(models_dir), pool_name="default") == ["org/default"]
    assert cli.load_bundle("shared", bundles_dir, str(models_dir), pool_name="customer-a") == ["org/tenant"]

    deps = collect_bundle_deps("shared", bundles_dir, models_dir, pool_name="customer-a")
    assert deps.models == ["org/tenant"]


def test_serve_filters_models_using_baked_bundle_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml")
    _write_model(models_dir / "supported.yaml", sie_id="org/supported")
    (models_dir / "unsupported.yaml").write_text(
        yaml.safe_dump(
            {
                "sie_id": "org/unsupported",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.unsupported:UnsupportedAdapter",
                    }
                },
            }
        )
    )
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir)],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].model_filter == ["org/supported"]
    assert "Bundle 'default': 1 models" in result.output


def test_serve_rejects_explicit_models_incompatible_with_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_bundle(
        bundles_dir / "transformers5.yaml",
        name="transformers5",
        adapters=["sie_server.adapters.transformers5"],
        deps={"transformers": ">=5"},
    )
    _write_model(models_dir / "supported.yaml", sie_id="org/supported")
    _write_model(
        models_dir / "selected.yaml",
        sie_id="org/selected",
        adapter="sie_server.adapters.transformers5:Transformers5Adapter",
    )
    called = False

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        [
            "serve",
            "--device",
            "cpu",
            "--models-dir",
            str(models_dir),
            "--models",
            "org/supported,org/selected",
        ],
    )

    assert result.exit_code == 1
    assert "org/selected" in result.output
    assert "baked bundle 'default'" in result.output
    assert "use an image whose bundle contains all selected models" in result.output
    assert not called
    assert os.environ["SIE_BUNDLE"] == "default"


def test_serve_allows_explicit_model_from_dependency_free_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(
        bundles_dir / "default.yaml",
        name="default",
        adapters=["sie_server.adapters.default"],
        deps={"transformers": ">=4.57,<5"},
    )
    _write_bundle(bundles_dir / "fake.yaml", name="fake")
    _write_model(models_dir / "selected.yaml", sie_id="sie-fake")
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir), "--models", "sie-fake"],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].model_filter == ["sie-fake"]
    assert os.environ["SIE_BUNDLE"] == "default"


def test_serve_allows_explicit_models_compatible_with_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_model(models_dir / "selected.yaml", sie_id="org/selected")
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir), "--models", "org/selected"],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].model_filter == ["org/selected"]
    assert os.environ["SIE_BUNDLE"] == "default"


def test_explicit_models_without_baked_bundle_remain_adhoc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_model(models_dir / "selected.yaml", sie_id="org/selected")
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir), "--models", "org/selected"],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].model_filter == ["org/selected"]
    assert os.environ["SIE_BUNDLE"].startswith("adhoc-")


def test_serve_rejects_explicit_bundle_different_from_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_bundle(bundles_dir / "transformers5.yaml", name="transformers5")
    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir), "--bundle", "transformers5"],
    )

    assert result.exit_code == 1
    assert "Bundle 'transformers5' is incompatible with baked bundle 'default'" in result.output


def test_serve_rejects_incompatible_extra_model_for_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_bundle(
        bundles_dir / "transformers5.yaml",
        name="transformers5",
        adapters=["sie_server.adapters.transformers5"],
        deps={"transformers": ">=5"},
    )
    _write_model(models_dir / "supported.yaml", sie_id="org/supported")
    _write_model(
        models_dir / "extra.yaml",
        sie_id="org/extra",
        adapter="sie_server.adapters.transformers5:Transformers5Adapter",
    )
    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setenv("SIE_EXTRA_MODELS", "org/extra")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", str(models_dir)],
    )

    assert result.exit_code == 1
    assert "org/extra" in result.output
    assert "baked bundle 'default'" in result.output


def test_serve_filters_cloud_models_using_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_bundle(
        bundles_dir / "transformers5.yaml",
        name="transformers5",
        adapters=["sie_server.adapters.transformers5"],
        deps={"transformers": ">=5"},
    )
    _write_model(models_dir / "supported.yaml", sie_id="org/supported")
    _write_model(
        models_dir / "lighton.yaml",
        sie_id="org/lighton",
        adapter="sie_server.adapters.transformers5:Transformers5Adapter",
    )
    cloud_configs = cli.load_model_configs(models_dir)
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **kwargs: Any) -> None:
        captured["config"] = config

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "load_model_configs", lambda models_dir: cloud_configs)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    result = CliRunner().invoke(
        cli.app,
        ["serve", "--device", "cpu", "--models-dir", "s3://bucket/models"],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].model_filter == ["org/supported"]
    assert os.environ["SIE_BUNDLE"] == "default"


def test_serve_rejects_incompatible_cloud_model_for_baked_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_unrelated_serve_env(monkeypatch)
    bundles_dir = tmp_path / "bundles"
    models_dir = tmp_path / "models"
    bundles_dir.mkdir()
    models_dir.mkdir()
    _write_bundle(bundles_dir / "default.yaml", name="default")
    _write_bundle(
        bundles_dir / "transformers5.yaml",
        name="transformers5",
        adapters=["sie_server.adapters.transformers5"],
        deps={"transformers": ">=5"},
    )
    _write_model(
        models_dir / "lighton.yaml",
        sie_id="org/lighton",
        adapter="sie_server.adapters.transformers5:Transformers5Adapter",
    )
    cloud_configs = cli.load_model_configs(models_dir)

    monkeypatch.setenv("SIE_BUNDLE", "default")
    monkeypatch.setattr(cli, "_DEFAULT_BUNDLES_DIR", bundles_dir)
    monkeypatch.setattr(cli, "load_model_configs", lambda models_dir: cloud_configs)

    result = CliRunner().invoke(
        cli.app,
        [
            "serve",
            "--device",
            "cpu",
            "--models-dir",
            "s3://bucket/models",
            "--models",
            "org/lighton",
        ],
    )

    assert result.exit_code == 1
    assert "org/lighton" in result.output
    assert "use an image built for bundle 'transformers5'" in result.output

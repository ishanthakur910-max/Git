from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import AdapterOptions, ModelConfig, ProfileConfig, Tasks
from sie_server.core.model_loader import ModelLoader
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry


@pytest.mark.parametrize(
    ("base_source", "expected_refs"),
    [
        pytest.param(
            {"hf_id": "acme/base", "hf_revision": "a" * 40},
            [("acme/base", "a" * 40), ("acme/draft", "b" * 40)],
            id="hf-base",
        ),
        pytest.param(
            {"weights_path": Path("/models/base")},
            [("acme/draft", "b" * 40)],
            id="local-base",
        ),
    ],
)
def test_ensure_weights_cached_stages_pinned_speculative_draft(
    base_source: dict[str, object],
    expected_refs: list[tuple[str, str]],
) -> None:
    draft_revision = "b" * 40
    config = ModelConfig(
        sie_id="acme/base",
        **base_source,
        tasks=Tasks(),
        profiles={
            "default": ProfileConfig(
                adapter_path="mod:Cls",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(
                    loadtime={
                        "speculative": {
                            "enabled": True,
                            "algorithm": "nextn",
                            "draft_model": "acme/draft",
                            "draft_model_revision": draft_revision,
                        }
                    }
                ),
            )
        },
    )
    cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-cpu")
    loader = ModelLoader(
        preprocessor_registry=PreprocessorRegistry(),
        postprocessor_registry=PostprocessorRegistry(cpu_pool),
        all_configs={},
    )
    cache_config = MagicMock()

    try:
        with (
            patch("sie_sdk.cache.get_cache_config", return_value=cache_config),
            patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/cache/model")) as ensure,
        ):
            loader.ensure_weights_cached(config)

        assert [(entry.args[0], entry.kwargs["revision"]) for entry in ensure.call_args_list] == expected_refs
        assert all(entry.args[1] is cache_config for entry in ensure.call_args_list)
    finally:
        loader._load_executor.shutdown(wait=True)
        cpu_pool.shutdown(wait=True)

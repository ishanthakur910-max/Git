from typing import Any

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class SGLangCuda13Adapter(SGLangGenerationAdapter):
    """Route generic generation through the CUDA 13 SGLang bundle."""


class SGLangStrictThinkingAdapter(SGLangCuda13Adapter):
    """CUDA 13 SGLang lane that guarantees a closed private thought block.

    SGLang 0.5.13 can suppress premature EOS tokens until the model emits its
    reasoning terminator.  The older shared SGLang bundle does not expose this
    launch option, so reasoning profiles that need the guarantee route through
    the existing CUDA 13 bundle while retaining the generic generation adapter.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        extra_launch_args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        launch_args = list(extra_launch_args or [])
        if "--enable-strict-thinking" not in launch_args:
            launch_args.append("--enable-strict-thinking")
        super().__init__(
            model_name_or_path,
            extra_launch_args=launch_args,
            **kwargs,
        )

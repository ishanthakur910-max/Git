from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# huggingface_hub's stock defaults (10 s for both DOWNLOAD and ETAG) are
# aggressive enough to spuriously fail on flaky home networks. The values
# here are tuned for slow-but-real connections: long enough that a 1
# Mbit/s link with brief stalls still completes a multi-GB download, short
# enough that a wedged socket fails in under a minute rather than hanging
# the load executor indefinitely.
#
# Operators override by setting the env var explicitly before launching
# the server — ``os.environ.setdefault`` is intentionally non-clobbering.
_DEFAULTS: dict[str, str] = {
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",  # per-chunk socket inactivity
    "HF_HUB_ETAG_TIMEOUT": "30",  # metadata HEAD request
}


def set_hf_default_timeouts() -> None:
    """Set sensible ``huggingface_hub`` timeout defaults if unset.

    Must be called BEFORE ``huggingface_hub`` is imported anywhere in the
    process — the library reads these env vars at module import time and
    caches the values. The server entry points (``cli.py``,
    ``main._create_app_from_env``) call this at the very top, before any
    transitive import that might pull in ``huggingface_hub``.

    Idempotent: subsequent calls are a no-op once the env vars are set.
    """
    for key, value in _DEFAULTS.items():
        if os.environ.setdefault(key, value) == value:
            logger.debug("set %s=%s (default; override by exporting before startup)", key, value)

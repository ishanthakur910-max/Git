from typing import Literal

OverflowPolicy = Literal["default", "truncate_text", "error"]
DEFAULT_OVERFLOW_POLICY: OverflowPolicy = "default"
VALID_OVERFLOW_POLICIES: frozenset[OverflowPolicy] = frozenset({"default", "truncate_text", "error"})

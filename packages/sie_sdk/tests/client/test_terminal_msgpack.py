from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from msgpack.exceptions import UnpackException
from sie_sdk import RequestError, ServerError, SIEAsyncClient, SIEClient
from sie_sdk._msgpack import packb as pack_msgpack
from sie_sdk.client._shared import get_error_detail, handle_error, parse_terminal_msgpack_object
from sie_sdk.client.async_ import _AioResponse

_ATTACKER_BODY = b"attacker-private-response-token"


def _object_array_sentinel() -> dict[bytes, object]:
    return {
        b"nd": True,
        b"type": "|O",
        b"kind": b"O",
        b"shape": [1],
        b"data": b"attacker-controlled-pickle",
    }


def _assert_content_safe_error(error: RequestError, *, expected: str, status_code: int, request_id: str) -> None:
    message = str(error)
    assert message == expected
    assert _ATTACKER_BODY.decode() not in message
    assert error.__context__ is None
    assert error.status_code == status_code
    assert error.request == {"id": request_id}


@pytest.mark.parametrize(
    ("status_code", "expected_prefix"),
    [
        (303, "Unexpected score HTTP response"),
        (200, "Malformed score MessagePack response"),
    ],
)
def test_sync_score_rejects_nonterminal_msgpack_content_safely(status_code: int, expected_prefix: str) -> None:
    request_id = f"req-sync-score-{status_code}"
    response = MagicMock(
        status_code=status_code,
        content=_ATTACKER_BODY,
        headers={"content-type": "application/msgpack", "x-sie-request-id": request_id},
    )

    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = response
        client = SIEClient("https://gateway.example.test")
        try:
            with pytest.raises(RequestError) as exc_info:
                client.score("Qwen/Qwen3-Reranker-4B", "query", ["candidate"])
        finally:
            client.close()

    _assert_content_safe_error(
        exc_info.value,
        expected=(
            f"{expected_prefix} "
            f"(status={status_code}, content_type=application/msgpack, body_bytes={len(_ATTACKER_BODY)})"
        ),
        status_code=status_code,
        request_id=request_id,
    )
    assert client_cls.call_args.kwargs["follow_redirects"] is False
    client_cls.return_value.post.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_prefix"),
    [
        (303, "Unexpected score HTTP response"),
        (200, "Malformed score MessagePack response"),
    ],
)
async def test_async_score_rejects_nonterminal_msgpack_content_safely(
    status_code: int,
    expected_prefix: str,
) -> None:
    request_id = f"req-async-score-{status_code}"
    response = _AioResponse(
        status_code,
        _ATTACKER_BODY,
        {"content-type": "application/msgpack", "x-sie-request-id": request_id},
    )
    client = SIEAsyncClient("https://gateway.example.test")
    client._post = AsyncMock(return_value=response)  # type: ignore[method-assign]
    try:
        with pytest.raises(RequestError) as exc_info:
            await client.score("Qwen/Qwen3-Reranker-4B", "query", ["candidate"])
    finally:
        await client.close()

    _assert_content_safe_error(
        exc_info.value,
        expected=(
            f"{expected_prefix} "
            f"(status={status_code}, content_type=application/msgpack, body_bytes={len(_ATTACKER_BODY)})"
        ),
        status_code=status_code,
        request_id=request_id,
    )
    client._post.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.parametrize("status_code", [200, 303])
def test_terminal_msgpack_never_unpickles_object_arrays(status_code: int) -> None:
    response = MagicMock(
        status_code=status_code,
        content=pack_msgpack({"items": [{"dense": {"values": _object_array_sentinel()}}]}),
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-object-array"},
    )

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        with pytest.raises(RequestError) as exc_info:
            parse_terminal_msgpack_object(response, owner="encode")

    pickle_loads.assert_not_called()
    assert exc_info.value.status_code == status_code
    assert exc_info.value.request == {"id": "req-object-array"}
    assert exc_info.value.__context__ is None
    assert b"attacker-controlled-pickle" not in str(exc_info.value).encode()


def test_msgpack_unpack_exceptions_are_sanitized() -> None:
    terminal_response = MagicMock(
        status_code=200,
        content=b"",
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-truncated"},
    )
    error_response = MagicMock(
        status_code=500,
        content=b"",
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-truncated"},
        text="safe fallback",
    )

    with patch(
        "sie_sdk.client._shared.unpack_msgpack",
        side_effect=UnpackException("private parser context"),
    ) as unpack:
        with pytest.raises(RequestError) as terminal_error:
            parse_terminal_msgpack_object(terminal_response, owner="encode")
        assert get_error_detail(error_response) is None
        with pytest.raises(ServerError, match="safe fallback") as server_error:
            handle_error(error_response)

    assert unpack.call_count == 3
    assert terminal_error.value.__context__ is None
    assert "private parser context" not in str(terminal_error.value)
    assert "private parser context" not in str(server_error.value)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(400, RequestError), (500, ServerError)],
)
def test_msgpack_error_envelopes_never_unpickle_object_arrays(
    status_code: int,
    error_type: type[RequestError | ServerError],
) -> None:
    response = MagicMock(
        status_code=status_code,
        content=pack_msgpack({"error": _object_array_sentinel()}),
        headers={"content-type": "application/msgpack"},
    )

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        detail = get_error_detail(response)
        with pytest.raises(error_type):
            handle_error(response)

    pickle_loads.assert_not_called()
    assert detail == _object_array_sentinel()

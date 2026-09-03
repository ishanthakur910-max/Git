"""Python client -> Rust local-ingest generation protocol tests."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgpack
import pytest
from sie_server import local_ingest_client


def _frame(value: dict[str, Any]) -> bytes:
    payload = msgpack.packb(value, use_bin_type=True)
    return len(payload).to_bytes(4, "little") + payload


async def _request(reader: asyncio.StreamReader) -> dict[str, Any]:
    length = int.from_bytes(await reader.readexactly(4), "little")
    return msgpack.unpackb(await reader.readexactly(length), raw=False)


def _meta() -> dict[str, Any]:
    return {
        "lane": "default|a100-80gb|test/model",
        "request_id": "req-1",
        "endpoint": "generate",
        "model": "test/model",
        "engine": "sglang",
        "admission_pool": "default",
        "bundle_config_hash": "hash",
    }


@pytest.fixture
def unix_socket_dir() -> Iterator[Path]:
    # macOS limits AF_UNIX paths to 104 bytes; pytest's tmp_path can exceed it.
    with tempfile.TemporaryDirectory(prefix="sie-li-", dir="/tmp") as directory:
        yield Path(directory)


def test_generation_body_preserves_gateway_transport_binding() -> None:
    meta = {
        **_meta(),
        "dispatch_context": b"authenticated-context",
        "payload_digest": b"d" * local_ingest_client.PAYLOAD_DIGEST_BYTES,
        "timeout_ms": 123,
    }

    body = local_ingest_client.build_generation_request_body(b"items", b"params", meta)

    assert body["dispatch_context"] == meta["dispatch_context"]
    assert body["payload_digest"] == meta["payload_digest"]
    assert body["timeout_ms"] == 123


@pytest.mark.asyncio
async def test_read_frame_normalizes_truncated_and_invalid_protocol_errors() -> None:
    truncated = asyncio.StreamReader()
    truncated.feed_data(b"\x01\x00")
    truncated.feed_eof()
    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="header is truncated"):
        await local_ingest_client._read_frame(truncated)

    invalid = asyncio.StreamReader()
    invalid.feed_data((1).to_bytes(4, "little") + b"\xc1")
    invalid.feed_eof()
    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="invalid MessagePack"):
        await local_ingest_client._read_frame(invalid)


@pytest.mark.asyncio
async def test_stream_generate_bounds_connect_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def never_connect(_socket_path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(local_ingest_client, "LOCAL_INGEST_CONNECT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(local_ingest_client.asyncio, "open_unix_connection", never_connect)

    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="connection timed out"):
        _ = [chunk async for chunk in local_ingest_client.stream_generate("unused.sock", b"items", b"params", _meta())]


@pytest.mark.asyncio
async def test_stream_generate_bounds_response_wait(unix_socket_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    socket_path = unix_socket_dir / "timeout.sock"

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        await asyncio.sleep(0.1)
        writer.close()
        await writer.wait_closed()

    monkeypatch.setattr(local_ingest_client, "DEFAULT_LOCAL_INGEST_RESPONSE_IDLE_TIMEOUT_S", 0.01)
    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        with pytest.raises(local_ingest_client.LocalIngestStreamError, match="response timed out"):
            _ = [
                chunk
                async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
            ]


@pytest.mark.asyncio
async def test_stream_generate_preserves_binding_order_and_terminal(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "ingest.sock"
    observed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await _request(reader)
        observed.set_result(request)
        writer.write(
            _frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"a", "seq": 0}})
            + _frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"b", "seq": 1}})
            + _frame(
                {
                    "id": 1,
                    "ok": True,
                    "error": None,
                    "body": {"final": True, "outcome": {"status": "complete", "chunks": 2}},
                }
            )
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        chunks = [
            chunk async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
        ]

    assert chunks == [b"a", b"b"]
    request = await observed
    assert request["op"] == "publish_generate_stream"
    body = request["body"]
    assert body["request_id"] == "req-1"
    assert body["payload_digest"] == local_ingest_client.compute_payload_digest(body)


@pytest.mark.asyncio
async def test_stream_generate_disconnect_cancels_connection_owned_stream(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "cancel.sock"
    disconnected: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        writer.write(_frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"first", "seq": 0}}))
        await writer.drain()
        disconnected.set_result(await reader.read())
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        stream = local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
        assert await anext(stream) == b"first"
        await stream.aclose()
        assert await asyncio.wait_for(disconnected, timeout=1.0) == b""


@pytest.mark.asyncio
async def test_stream_generate_rejects_transport_sequence_gap(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "gap.sock"

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        writer.write(_frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"late", "seq": 1}}))
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        with pytest.raises(local_ingest_client.LocalIngestStreamError, match="not expected 0"):
            _ = [
                chunk
                async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
            ]

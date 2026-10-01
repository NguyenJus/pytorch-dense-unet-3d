"""Safe pidfd stop works even when Python lacks Linux-specific bindings."""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import subprocess
import sys

import pytest

from dense_unet_3d.training import runtime


@pytest.fixture()
def without_python_pidfds(monkeypatch):
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)


def test_libc_stop_controlled_child(tmp_path, without_python_pidfds, monkeypatch):
    opened = []
    actual_open = runtime._pidfd_open

    def track_open(pid):
        fd = actual_open(pid)
        opened.append(fd)
        return fd

    monkeypatch.setattr(runtime, "_pidfd_open", track_open)
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            """
import signal
import sys
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(23))
print('ready', flush=True)
signal.pause()
""",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], "child did not become ready"
        assert child.stdout.readline().strip() == "ready"
        runtime.atomic_json(
            tmp_path / "runtime.json",
            {
                "owner": runtime._process_identity(child.pid),
                "terminal_reason": None,
            },
        )
        runtime.request_stop(tmp_path)
        assert child.wait(timeout=10) == 23
        assert len(opened) == 1
        with pytest.raises(OSError) as error:
            os.fstat(opened[0])
        assert error.value.errno == errno.EBADF
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if child.stdout is not None:
            child.stdout.close()


def test_stale_identity_refused_before_pidfd_open(tmp_path, monkeypatch):
    stale = runtime._process_identity(os.getpid())
    stale["start_ticks"] = "-1"
    runtime.atomic_json(tmp_path / "runtime.json", {"owner": stale, "terminal_reason": None})
    monkeypatch.setattr(
        runtime, "_pidfd_open", lambda _pid: pytest.fail("must not open stale owner")
    )
    with pytest.raises(RuntimeError, match="no verified live owner"):
        runtime.request_stop(tmp_path)


@pytest.mark.parametrize("failure", ["identity_changed", "process_disappeared", "send_failed"])
def test_pidfd_closed_on_recheck_and_send_failure(tmp_path, monkeypatch, failure):
    owner = runtime._process_identity(os.getpid())
    monkeypatch.setattr(runtime, "read_status", lambda _path: {"owner": owner, "ownership": "live"})
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(runtime, "_pidfd_open", lambda _pid: fd)

    class LivePoll:
        def register(self, *_args):
            pass

        def poll(self, _timeout):
            return []

    monkeypatch.setattr(runtime.select, "poll", LivePoll)
    if failure == "identity_changed":
        monkeypatch.setattr(
            runtime, "_process_identity", lambda _pid: {**owner, "start_ticks": "-1"}
        )
    elif failure == "process_disappeared":

        def disappeared(_pid):
            raise FileNotFoundError("process disappeared")

        monkeypatch.setattr(runtime, "_process_identity", disappeared)
    else:

        def denied(_fd, _sig):
            raise PermissionError(errno.EPERM, "permission denied")

        monkeypatch.setattr(runtime, "_pidfd_send_signal", denied)
    if failure != "send_failed":
        monkeypatch.setattr(
            runtime, "_pidfd_send_signal", lambda *_: pytest.fail("must not signal changed owner")
        )
    with pytest.raises((RuntimeError, OSError)):
        runtime.request_stop(tmp_path)
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


def test_libc_errno_propagates(without_python_pidfds):
    with pytest.raises(OSError) as open_error:
        runtime._pidfd_open(-1)
    assert open_error.value.errno == errno.EINVAL
    with pytest.raises(OSError) as send_error:
        runtime._pidfd_send_signal(-1, signal.SIGTERM)
    assert send_error.value.errno == errno.EBADF


@pytest.mark.parametrize("operation", ["open", "send"])
def test_missing_libc_symbol_fails_safely(monkeypatch, without_python_pidfds, operation):
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: object())
    with pytest.raises(RuntimeError, match="Safe stop unavailable"):
        if operation == "open":
            runtime._pidfd_open(os.getpid())
        else:
            runtime._pidfd_send_signal(0, signal.SIGTERM)


def test_native_pidfd_bindings_preferred(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "pidfd_open", lambda *args: calls.append(args) or 123, raising=False)
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda *args: calls.append(args), raising=False
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: pytest.fail("libc not needed"))
    assert runtime._pidfd_open(456) == 123
    runtime._pidfd_send_signal(123, signal.SIGTERM)
    assert calls == [(456, 0), (123, signal.SIGTERM, None, 0)]

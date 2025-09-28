"""Pytest fixtures for spinning up the Streamlit app during UI tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _wait_for_healthy(port: int, timeout: float = 60.0) -> str:
    base_url = f"http://127.0.0.1:{port}"
    health_url = f"{base_url}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return base_url
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError("Streamlit app did not become healthy in time.")


@pytest.fixture(scope="session")
def app_base_url() -> str:
    port = _find_open_port()
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "")
    env.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    env.setdefault("OPENAI_IMAGE_MODEL", "dall-e-3")
    env.setdefault("OPENAI_IMAGE_SIZE", "512x512")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    try:
        base_url = _wait_for_healthy(port)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

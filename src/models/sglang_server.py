"""SGLang server lifecycle management via subprocess.

Handles launching, health-checking, and killing the SGLang server so it
doesn't conflict with DAPO training for GPU memory. The server is only
alive during evaluation — never during training.

Usage:
    server = SGLangServer("Qwen/Qwen3-4B")
    server.start()          # launches subprocess, waits for healthy
    # ... run evaluation ...
    server.stop()           # SIGTERM → SIGKILL, frees GPU memory

    # Context manager for automatic cleanup:
    with SGLangServer("Qwen/Qwen3-4B") as server:
        url = server.url    # "http://127.0.0.1:30000"
        # ... evaluate ...
    # server killed automatically even on exception
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


class SGLangServer:
    """Managed SGLang server subprocess with proper lifecycle."""

    def __init__(
        self,
        model_name: str,
        host: str = "127.0.0.1",
        port: int = 30000,
        mem_fraction: float = 0.85,
        max_wait: int = 300,
        log_dir: str = "results/logs",
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.mem_fraction = mem_fraction
        self.max_wait = max_wait
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen | None = None
        self._registered_cleanup = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def is_healthy(self) -> bool:
        """Check if the server is up and responding."""
        try:
            import requests
            resp = requests.get(f"{self.url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        """Launch SGLang server as a subprocess and wait for it to be healthy.

        Returns True if server started successfully, False on failure.
        Kills any existing server on the same port first.
        """
        if self.is_healthy:
            logger.info(f"SGLang already running at {self.url}")
            return True

        self._kill_existing_on_port()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"sglang_{self.port}.log"

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--mem-fraction-static", str(self.mem_fraction),
            "--trust-remote-code",
            "--log-level", "warning",
        ]

        logger.info(f"Starting SGLang: {self.model_name} on {self.url}")
        logger.info(f"  Log: {log_file}")

        with open(log_file, "w") as lf:
            self.process = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

        if not self._registered_cleanup:
            atexit.register(self.stop)
            self._registered_cleanup = True

        if not self._wait_for_healthy():
            logger.error(f"SGLang failed to start within {self.max_wait}s")
            self._dump_log_tail(log_file)
            self.stop()
            return False

        logger.success(f"SGLang ready at {self.url} (PID: {self.process.pid})")
        return True

    def stop(self):
        """Gracefully stop the SGLang server, freeing GPU memory."""
        if self.process is None:
            return

        pid = self.process.pid
        if self.process.poll() is not None:
            logger.info(f"SGLang (PID {pid}) already exited")
            self.process = None
            return

        logger.info(f"Stopping SGLang (PID {pid})...")

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self.process.wait(timeout=10)
            logger.info(f"SGLang (PID {pid}) stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(f"SGLang (PID {pid}) didn't stop, sending SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                self.process.wait(timeout=5)
            except Exception:
                pass

        self.process = None

        time.sleep(2)

    def restart(
        self,
        model_name: str | None = None,
        mem_fraction: float | None = None,
    ) -> bool:
        """Stop the server and start again, optionally with a different model or memory fraction."""
        self.stop()
        if model_name:
            self.model_name = model_name
        if mem_fraction is not None:
            self.mem_fraction = mem_fraction
        return self.start()

    def _wait_for_healthy(self) -> bool:
        """Poll the health endpoint until ready or timeout."""
        waited = 0
        interval = 3
        while waited < self.max_wait:
            if self.process.poll() is not None:
                logger.error(f"SGLang process exited with code {self.process.returncode}")
                return False
            if self.is_healthy:
                return True
            time.sleep(interval)
            waited += interval
            if waited % 15 == 0:
                logger.info(f"  Waiting for SGLang... ({waited}s / {self.max_wait}s)")
        return False

    def _kill_existing_on_port(self):
        """Kill any process already listening on our port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed existing process on port {self.port} (PID {pid})")
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(2)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _dump_log_tail(self, log_file: Path, n_lines: int = 20):
        """Print the last N lines of the server log for debugging."""
        try:
            lines = log_file.read_text().strip().split("\n")
            tail = lines[-n_lines:]
            logger.error("SGLang server log (last lines):")
            for line in tail:
                logger.error(f"  {line}")
        except Exception:
            pass

    def __enter__(self):
        if not self.start():
            raise RuntimeError(
                f"Failed to start SGLang server for {self.model_name} on {self.url}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

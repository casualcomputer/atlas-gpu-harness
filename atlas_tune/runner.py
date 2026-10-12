"""Start, watch and stop engine containers. Only containers labelled by atlas-tune are touched."""

from __future__ import annotations

import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx

from . import hardware
from .engines import Engine, docker_run
from .models import Artifact

LABEL = "atlas-tune"


class EngineFailed(RuntimeError):
    def __init__(self, message: str, logs: str = ""):
        super().__init__(message)
        self.logs = logs


@dataclass
class Server:
    name: str
    base_url: str
    model: str = ""
    argv: list[str] = field(default_factory=list)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GpuSampler:
    """Samples total GPU memory in use while a trial runs (includes other processes)."""

    def __init__(self, interval: float = 0.5):
        self.interval, self.peak_gib, self._stop = interval, None, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            used = [g["used_gib"] for g in hardware.gpus()]
            if used:
                self.peak_gib = max(self.peak_gib or 0, sum(used))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


class DockerRunner:
    def __init__(self, log=print):
        self.log = log
        self._pulled: dict[str, str] = {}

    def _docker(self, *args, timeout=60, check=True) -> str:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout,
                              check=check).stdout.strip()

    def image_id(self, engine: Engine) -> str:
        """Pull once per run and return the immutable digest, so memory keys on the exact version."""
        if engine.image not in self._pulled:
            self.log(f"  pulling {engine.image} (first time can take several minutes)")
            self._docker("pull", "-q", engine.image, timeout=3600)
            digest = self._docker("image", "inspect", "--format",
                                  "{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}", engine.image)
            self._pulled[engine.image] = digest
        return self._pulled[engine.image]

    def start(self, engine: Engine, artifact: Artifact, config: dict, context: int) -> Server:
        port, name = free_port(), f"{LABEL}-{engine.name}-{uuid.uuid4().hex[:8]}"
        argv = docker_run(engine, artifact, config, context, host_port=port, name=name, labels={LABEL: "1"})
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise EngineFailed(f"docker run failed: {result.stderr.strip()[-500:]}")
        return Server(name=name, base_url=f"http://127.0.0.1:{port}", argv=argv)

    def running(self, server: Server) -> bool:
        return self._docker("inspect", "-f", "{{.State.Running}}", server.name, check=False) == "true"

    def logs(self, server: Server, lines: int = 80) -> str:
        result = subprocess.run(["docker", "logs", "--tail", str(lines), server.name],
                                capture_output=True, text=True, timeout=30)
        return (result.stdout + result.stderr)[-6000:]

    def wait_ready(self, server: Server, timeout: float = 1800) -> Server:
        """Ready = /v1/models answers. Generous timeout: the first start downloads weights."""
        start = last_note = time.monotonic()
        with httpx.Client(timeout=5, trust_env=False) as client:
            while time.monotonic() - start < timeout:
                if not self.running(server):
                    raise EngineFailed("container exited during startup", self.logs(server))
                try:
                    response = client.get(server.base_url + "/v1/models")
                    if response.status_code == 200 and response.json().get("data"):
                        server.model = response.json()["data"][0]["id"]
                        return server
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() - last_note > 60:
                    last_note = time.monotonic()
                    self.log(f"  still starting ({int(last_note - start)} s; downloading or loading weights)")
                time.sleep(2)
        raise EngineFailed(f"not ready after {timeout:.0f} s", self.logs(server))

    def stop(self, server: Server) -> None:
        subprocess.run(["docker", "rm", "-f", server.name], capture_output=True, timeout=120)

    def cleanup(self) -> int:
        """Remove leftover trial containers from an interrupted run (ours only, by label)."""
        ids = self._docker("ps", "-aq", "--filter", f"label={LABEL}=1", check=False).split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=120)
        return len(ids)

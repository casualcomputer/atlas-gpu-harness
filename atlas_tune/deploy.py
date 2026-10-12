"""Turn a verified trial into deployable files: serve.sh, compose.yaml and k8s.yaml.

All three pin the engine image by the digest that was measured, require an API key, and
use the same engine arguments as the winning trial (at the context length it was tested).
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from .engines import CA_PATH, ENGINES, PROXY_VARS, docker_run
from .models import Artifact

PORT = 8000


def _parts(trial: dict):
    engine = ENGINES[trial["engine"]]
    artifact = Artifact(**trial["artifact"])
    image = trial.get("image") or engine.image
    args = engine.args(artifact, trial["config"], trial["context"])  # bridge networking: engine.port
    return engine, artifact, image, args


def _q(value) -> str:
    return json.dumps(str(value))  # JSON strings are valid YAML scalars


def serve_sh(trial: dict, run_id: str, model: str) -> str:
    engine, artifact, image, _ = _parts(trial)
    argv = docker_run(engine, artifact, trial["config"], trial["context"], host_port=PORT,
                      name=f"{engine.name}-serve", production=True)
    argv[argv.index(engine.image)] = image  # pin the exact digest that was measured
    return ("#!/usr/bin/env bash\n"
            f"# {model} on {engine.name}, verified by atlas-tune run {run_id}\n"
            "# Clients send: Authorization: Bearer $API_KEY\n"
            "set -euo pipefail\n"
            ': "${API_KEY:?export API_KEY=<a long random secret> first}"\n'
            f'{shlex.join(argv)} --api-key "$API_KEY"\n'
            f'echo "Serving on http://localhost:{PORT}/v1 (container {engine.name}-serve)"\n')


def compose_yaml(trial: dict, model: str) -> str:
    engine, _, image, args = _parts(trial)
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    llama_cache = Path(os.environ.get("LLAMA_CACHE", Path.home() / ".cache/llama.cpp"))
    command = "\n".join(f"      - {_q(a)}" for a in [*args, "--api-key", "${API_KEY:?set API_KEY}"])
    proxy_env = "".join(f"\n      - {name}" for name in PROXY_VARS if os.environ.get(name))
    ca_volume = ""
    if os.environ.get("ATLAS_CA_BUNDLE"):
        ca = Path(os.environ["ATLAS_CA_BUNDLE"]).expanduser().resolve()
        ca_volume = f"\n      - {_q(f'{ca}:{CA_PATH}:ro')}"
        proxy_env += "".join(f"\n      - {n}={CA_PATH}" for n in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"))
    gpu = "" if os.environ.get("ATLAS_GPUS") == "none" else """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]"""
    return f"""# {model} on {engine.name}, verified by atlas-tune. Run: API_KEY=... docker compose up -d
services:
  llm:
    image: {_q(image)}
    command:
{command}
    ports:
      - "{PORT}:{engine.port}"
    ipc: host
    security_opt:
      - no-new-privileges:true
    environment:
      - HF_TOKEN
      - LLAMA_CACHE=/root/.cache/llama.cpp{proxy_env}
    volumes:
      - {_q(f"{hf_cache}:/root/.cache/huggingface")}
      - {_q(f"{llama_cache}:/root/.cache/llama.cpp")}{ca_volume}
    restart: unless-stopped{gpu}
"""


def k8s_yaml(trial: dict, model: str) -> str:
    engine, _, image, args = _parts(trial)
    name = f"llm-{engine.name}"
    arg_lines = "\n".join(f"            - {_q(a)}" for a in [*args, "--api-key", "$(API_KEY)"])
    gpu_limit = "" if os.environ.get("ATLAS_GPUS") == "none" else """
          resources:
            limits:
              nvidia.com/gpu: 1"""
    probe = f"""httpGet:
              path: /health
              port: {engine.port}"""
    return f"""# {model} on {engine.name}, verified by atlas-tune on this GPU model.
# Before applying: kubectl create secret generic llm-api-key --from-literal=api-key=<secret>
# and a PersistentVolumeClaim named model-cache for weights (or change the volume below).
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: {name}}}
  template:
    metadata:
      labels: {{app: {name}}}
    spec:
      containers:
        - name: engine
          image: {_q(image)}
          args:
{arg_lines}
          ports:
            - containerPort: {engine.port}
          env:
            - name: API_KEY
              valueFrom: {{secretKeyRef: {{name: llm-api-key, key: api-key}}}}
            - name: HF_TOKEN
              valueFrom: {{secretKeyRef: {{name: hf-token, key: token, optional: true}}}}
            - name: LLAMA_CACHE
              value: /root/.cache/llama.cpp{gpu_limit}
          securityContext:
            allowPrivilegeEscalation: false
          startupProbe:
            {probe}
            periodSeconds: 10
            failureThreshold: 180
          readinessProbe:
            {probe}
            periodSeconds: 10
          livenessProbe:
            {probe}
            periodSeconds: 30
            failureThreshold: 3
          volumeMounts:
            - {{name: cache, mountPath: /root/.cache/huggingface, subPath: huggingface}}
            - {{name: cache, mountPath: /root/.cache/llama.cpp, subPath: llama.cpp}}
            - {{name: shm, mountPath: /dev/shm}}
      volumes:
        - name: cache
          persistentVolumeClaim: {{claimName: model-cache}}
        - name: shm
          emptyDir: {{medium: Memory, sizeLimit: 16Gi}}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector: {{app: {name}}}
  ports:
    - port: {PORT}
      targetPort: {engine.port}
"""


def write_all(directory: Path, trial: dict, run_id: str, model: str) -> list[Path]:
    files = {"serve.sh": serve_sh(trial, run_id, model), "compose.yaml": compose_yaml(trial, model),
             "k8s.yaml": k8s_yaml(trial, model)}
    paths = []
    for name, text in files.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    (directory / "serve.sh").chmod(0o755)
    return paths

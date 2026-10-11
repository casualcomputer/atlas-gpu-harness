#!/usr/bin/env python3
"""Install a pinned llama.cpp runtime locally; never install system packages or models."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11429"
ASSETS = {
    "llama.tar.gz": (
        f"{RELEASE}/llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "b13c64b9224d3c89d9993945d70cc21177ad00178c21b8d34237e3bc139834f1",
        "llama-b11429"),
    "cudart.tar.gz": (
        f"{RELEASE}/cudart-llama-b11429-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "f4ab593e31507320b121f2cb21791f4c72420a2483d3df8bf2ca1a97d04e6845",
        "cudart-llama-b11429-bin-ubuntu-cuda-12.8-x64"),
    "libc6_2.39-0ubuntu8.9_amd64.deb": (
        "https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8.9_amd64.deb",
        "ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be",
        "portable-libc"),
}
SHELL_PREFIX = '''#!/usr/bin/env bash
set -euo pipefail
engine_home="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
engine_libs="$engine_home/llama-b11429"
cuda_dir="$engine_home/cudart-llama-b11429-bin-ubuntu-cuda-12.8-x64"
unset GGML_BACKEND_PATH
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(base, name):
    url, expected, _ = ASSETS[name]
    target = base / name
    if not target.exists():
        partial = target.with_name(name + ".download")
        if not partial.exists() or sha256(partial) != expected:
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                            "--output", str(partial), url], check=True)
        if sha256(partial) != expected:
            raise ValueError(f"Checksum failed for {partial}; remove that partial file and retry")
        partial.replace(target)
    if sha256(target) != expected:
        raise ValueError(f"Checksum failed for {target}; existing file was left untouched")
    return target


def extract(base, name):
    archive = download(base, name)
    _, expected, folder = ASSETS[name]
    destination = base / folder
    marker = destination / ".atlas-archive-sha256"
    if destination.exists():
        if marker.is_file() and marker.read_text().strip() == expected:
            return destination
        raise ValueError(f"Unverified existing extraction {destination}; use a fresh --engine-dir")
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=base) as temporary:
        staging = Path(temporary)
        if name.endswith(".deb"):
            output = staging / folder
            subprocess.run(["dpkg-deb", "--extract", str(archive), str(output)], check=True)
        else:
            with tarfile.open(archive, "r:gz") as package:
                package.extractall(staging, filter="data")
            output = staging / folder
        if not output.is_dir():
            raise ValueError(f"Expected directory {folder} was absent from verified archive")
        (output / marker.name).write_text(expected + "\n")
        output.rename(destination)
    return destination


def command_result(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
    return result.returncode, (result.stdout + result.stderr).strip()


def verify_launcher(launcher):
    code, version = command_result([str(launcher), "--version"])
    if code or "11429" not in version or "d81235049" not in version:
        raise ValueError(f"Pinned engine version check failed: {version[-1200:]}")
    code, devices = command_result([str(launcher), "--list-devices"])
    if code or "CUDA0:" not in devices:
        raise ValueError(f"CUDA device check failed: {devices[-1200:]}")
    return {"version": version, "devices": devices}


def write_atomic(path, text):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        output.write(text)
        temporary = Path(output.name)
    temporary.replace(path)


def manifest(base, launcher, checks):
    assets = {}
    for name, (url, expected, _) in ASSETS.items():
        path = base / name
        if path.exists():
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Cached asset checksum failed: {path}")
            assets[name] = {"url": url, "sha256": actual, "bytes": path.stat().st_size}
    libraries = {}
    for path in sorted((base / "portable-cxx").glob("*.so*")):
        libraries[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {"release": "b11429", "commit": "d81235049", "assets": assets,
            "launcher": str(launcher), "launcher_sha256": sha256(launcher),
            "portable_cxx": libraries, "checks": checks,
            "scope": "Local inference runtime only; no model download or inference performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/engines/llama-b11429")
    parser.add_argument("--cxx-dir", type=Path, default=Path.home() / "miniconda3/lib",
                        help="Existing libstdc++.so.6 and libgcc_s.so.1 directory, needed on older Linux")
    parser.add_argument("--check", action="store_true", help="Verify an existing launcher without writing files")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("This pinned runtime supports Linux/WSL x86_64 only")
    if not hasattr(tarfile, "data_filter"):
        parser.error("Use a current Python with tarfile.data_filter (for example the project .venv)")
    base = args.engine_dir.expanduser().absolute()
    launcher = base / "launch-llama.sh"
    if launcher.exists():
        # Preserve an already working launcher, including the original local setup.
        result = manifest(base, launcher, verify_launcher(launcher))
        if not args.check:
            write_atomic(base / "setup-manifest.json", json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.check:
        parser.error(f"No launcher at {launcher}; run setup without --check first")
    if not shutil.which("curl"):
        parser.error("curl is required for resumable downloads")
    base.mkdir(parents=True, exist_ok=True)
    engine = extract(base, "llama.tar.gz")
    cuda = extract(base, "cudart.tar.gz")
    env = {**os.environ, "LD_LIBRARY_PATH": f"{engine}:{cuda}:/usr/lib/wsl/lib"}
    code, diagnostic = command_result([str(engine / "llama-server"), "--version"], env)
    if code == 0:
        shell = SHELL_PREFIX + '''export LD_LIBRARY_PATH="$engine_libs:$cuda_dir:/usr/lib/wsl/lib"
exec "$engine_libs/llama-server" "$@"
'''
    else:
        if "GLIBC" not in diagnostic:
            raise ValueError(f"Native runtime failed for a reason other than library versions: {diagnostic[-1200:]}")
        cxx = args.cxx_dir.expanduser()
        for name in ("libstdc++.so.6", "libgcc_s.so.1"):
            if not (cxx / name).is_file():
                raise ValueError(f"Provide --cxx-dir with modern {name} (GCC 13+), or use Ubuntu 24.04. "
                                 "No system packages were changed.")
        if not shutil.which("dpkg-deb"):
            raise ValueError("dpkg-deb is required to extract the local Ubuntu runtime")
        libc = extract(base, "libc6_2.39-0ubuntu8.9_amd64.deb") / "usr/lib/x86_64-linux-gnu"
        local_cxx = base / "portable-cxx"
        local_cxx.mkdir(exist_ok=True)
        for name in ("libstdc++.so.6", "libgcc_s.so.1"):
            if not (local_cxx / name).exists():
                shutil.copyfile(cxx / name, local_cxx / name)
        # Backends locate /proc/self/exe; keep the loader beside the engine's backend libraries.
        shutil.copy2(libc / "ld-linux-x86-64.so.2", engine / "ld-linux-x86-64.so.2")
        shell = SHELL_PREFIX + '''libc_dir="$engine_home/portable-libc/usr/lib/x86_64-linux-gnu"
exec "$engine_libs/ld-linux-x86-64.so.2" \\
  --library-path "$libc_dir:$engine_home/portable-cxx:$engine_libs:$cuda_dir:/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu" \\
  "$engine_libs/llama-server" "$@"
'''
    # Check a temporary launcher in the same directory before publishing it.
    trial = base / ".launch-llama.check.sh"
    write_atomic(trial, shell)
    trial.chmod(0o755)
    checks = verify_launcher(trial)
    trial.replace(launcher)
    result = manifest(base, launcher, checks)
    write_atomic(base / "setup-manifest.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Setup stopped: {error}") from error

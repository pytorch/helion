#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import suppress
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
import requests

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_ALLOWED_CUDA_VERSIONS = ["13.0"]
DEFAULT_DATA_CENTER_IDS = ["US-CA-2", "US-IL-1", "US-GA-2"]
DEFAULT_DATA_CENTER_PRIORITY = "availability"
DEFAULT_GPU_TYPE_IDS = ["NVIDIA B200"]
DEFAULT_IMAGE_NAME = "docker.io/jansel/helion:latest"
DEFAULT_PORTS = ["22/tcp"]
DEFAULT_API_URL = "https://rest.runpod.io/v1"
DEFAULT_REMOTE_DIR = "/workspace/helion"
DEFAULT_STATE_DIR = ".runpod-state"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_TIMEOUT = 20 * 60
DEFAULT_CREATE_RETRIES = 8
DEFAULT_CREATE_RETRY_INITIAL_DELAY = 15.0
DEFAULT_CREATE_RETRY_MAX_DELAY = 120.0
RSYNC_EXCLUDES = [
    ".benchmarks",
    ".claude",
    ".github",
    ".git",
    ".hypothesis",
    ".idea",
    ".mypy_cache",
    ".pyre",
    ".pytest_cache",
    ".runpod-state",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    "dist",
    "site",
]
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=4",
]
INLINE_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
SHELL_META_CHARS = set("|&;<>()$`\\*?[]{}~\n")
REMOTE_EXEC_PY = """
import json
import os
import sys


def select_shell(*, prefer_login: bool) -> tuple[str, bool]:
    login_candidates = [os.environ.get("SHELL"), "/bin/bash"]
    shell_candidates = [os.environ.get("SHELL"), "/bin/bash", "/bin/sh"]
    for candidate in (login_candidates if prefer_login else shell_candidates):
        if candidate and os.path.exists(candidate):
            if prefer_login and os.path.basename(candidate) == "sh":
                break
            return candidate, True
    return "/bin/sh" if os.path.exists("/bin/sh") else "sh", False


remote_dir, mode, payload_json, env_json = sys.argv[1:5]
os.chdir(remote_dir)
env = os.environ.copy()
env.update(json.loads(env_json))

if mode == "exec":
    argv = json.loads(payload_json)
    if not argv:
        raise SystemExit("remote exec argv is empty")
    os.execvpe(argv[0], argv, env)

shell, supports_login = select_shell(prefer_login=(mode == "login"))
shell_name = os.path.basename(shell) or shell
if mode == "shell":
    os.execvpe(shell, [shell_name, "-c", json.loads(payload_json)], env)
elif mode == "login":
    if not supports_login:
        os.execvpe(shell, [shell_name], env)
    os.execvpe(shell, [shell_name, "-l"], env)
else:
    raise SystemExit(f"unknown remote mode: {mode}")
""".strip()


class RunpodError(RuntimeError):
    pass


class RunpodApiError(RunpodError):
    def __init__(
        self,
        *,
        method: str,
        path: str,
        status: int | str,
        body: object,
    ) -> None:
        self.method = method
        self.path = path
        self.status = status
        self.body = body
        self.error_text = extract_api_error_text(body)
        super().__init__(
            f"Runpod API {method} {path} failed: HTTP {status}: {self.error_text}"
        )


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


class PodLock:
    def __init__(self, repo_root: Path, pod_id: str) -> None:
        self.repo_root = repo_root
        self.state_dir = repo_root / DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pod_id = pod_id
        self.lock_path = self.state_dir / f"{pod_id}.lock"
        self.lock = FileLock(str(self.lock_path))

    def acquire(self) -> None:
        try:
            self.lock.acquire(blocking=False)
        except FileLockTimeout as exc:
            raise RunpodError(
                f"Pod {self.pod_id} is already in use by another local process"
            ) from exc

    def release(self) -> None:
        try:
            self.lock.release()
        finally:
            with suppress(OSError):
                if self.lock_path.exists():
                    self.lock_path.unlink()


def slugify(text: str, max_len: int = 32) -> str:
    chars = []
    last_dash = False
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    value = "".join(chars).strip("-")
    return value[:max_len] or "x"


def compute_pod_name(repo_root: Path) -> str:
    username = slugify(getpass.getuser(), 24)
    hostname = slugify(socket.gethostname().split(".")[0], 24)
    digest = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:10]
    return f"helion-{username}-{hostname}-{digest}"[:191]


def read_runpod_config() -> dict[str, Any]:
    config_path = Path.home() / ".runpod" / "config.toml"
    if not config_path.exists():
        raise RunpodError(f"Runpod config not found: {config_path}")
    return tomllib.loads(config_path.read_text())


def pod_lock_path(repo_root: Path, pod_id: str) -> Path:
    return repo_root / DEFAULT_STATE_DIR / f"{pod_id}.lock"


def managed_pod_state_path(repo_root: Path, pod_name: str) -> Path:
    state_dir = repo_root / DEFAULT_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{slugify(pod_name, 96)}.json"


def read_managed_pod_state(repo_root: Path, pod_name: str) -> dict[str, Any] | None:
    path = managed_pod_state_path(repo_root, pod_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunpodError(f"Failed to read managed pod state {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunpodError(f"Unexpected managed pod state in {path}: {data!r}")
    return data


def write_managed_pod_state(repo_root: Path, pod_name: str, pod_id: str) -> None:
    path = managed_pod_state_path(repo_root, pod_name)
    path.write_text(
        json.dumps(
            {
                "name": pod_name,
                "pod_id": pod_id,
                "persistent": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def clear_managed_pod_state(repo_root: Path, pod_name: str) -> None:
    with suppress(OSError):
        managed_pod_state_path(repo_root, pod_name).unlink()


def extract_api_error_text(body: object) -> str:
    if isinstance(body, dict):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(body, sort_keys=True)
    if isinstance(body, str):
        return body.strip() or "<empty response body>"
    return repr(body)


def is_capacity_error(exc: Exception) -> bool:
    if not isinstance(exc, RunpodApiError):
        return False
    message = exc.error_text.lower()
    markers = (
        "there are no instances currently available",
        "no spot price found",
        "could not find any pods with required specifications",
        "insufficient capacity",
        "no capacity",
    )
    return any(marker in message for marker in markers)


def format_capacity_request(payload: dict[str, Any]) -> str:
    cloud = payload.get("cloudType", "SECURE")
    interruptible = payload.get("interruptible")
    gpu_types = ", ".join(payload.get("gpuTypeIds", [])) or "unspecified"
    cuda_versions = ", ".join(payload.get("allowedCudaVersions", [])) or "any"
    data_centers = ", ".join(payload.get("dataCenterIds", [])) or "any"
    data_center_priority = payload.get("dataCenterPriority") or "default"
    return (
        f"cloud={cloud} interruptible={interruptible} "
        f"gpu={gpu_types} cuda={cuda_versions} "
        f"dataCenters={data_centers} dataCenterPriority={data_center_priority}"
    )


def format_create_pod_error(
    payload: dict[str, Any],
    exc: Exception,
    *,
    attempts: int,
    max_attempts: int,
) -> RunpodError:
    if not is_capacity_error(exc):
        if isinstance(exc, RunpodError):
            return exc
        return RunpodError(str(exc))

    assert isinstance(exc, RunpodApiError)
    request_summary = format_capacity_request(payload)
    suggestions = [
        "retry in a minute",
        "increase --create-retries",
        "omit --allowed-cuda-version to broaden placement",
        "use --any-data-center to remove the default region allowlist",
        "add more --gpu-type-id values if you want hardware fallback",
    ]
    return RunpodError(
        "Runpod could not find capacity for the requested pod after "
        f"{attempts}/{max_attempts} create attempt(s): {request_summary}. "
        f"API said: {exc.error_text}. Try: {', '.join(suggestions)}."
    )


def load_api_settings(args: argparse.Namespace) -> tuple[str, str]:
    config = read_runpod_config()
    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY") or config.get("apikey")
    api_url = args.api_url or os.environ.get("RUNPOD_API_URL") or DEFAULT_API_URL
    if not api_key:
        raise RunpodError(
            "Runpod API key not found in --api-key, RUNPOD_API_KEY, or ~/.runpod/config.toml"
        )
    return api_key, api_url.rstrip("/")


def api_request(
    api_key: str,
    api_url: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | list[Any] | None = None,
) -> object:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.request(
            method,
            f"{api_url}{path}",
            headers=headers,
            params={
                key: value for key, value in (params or {}).items() if value is not None
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body.strip()
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RunpodApiError(
            method=method,
            path=path,
            status=status,
            body=parsed,
        ) from exc
    except requests.RequestException as exc:
        raise RunpodError(f"Runpod API {method} {path} failed: {exc}") from exc


def get_pods(
    api_key: str, api_url: str, include_machine: bool = True
) -> list[dict[str, Any]]:
    pods = api_request(
        api_key,
        api_url,
        "GET",
        "/pods",
        params={"includeMachine": str(include_machine).lower()},
    )
    if not isinstance(pods, list):
        raise RunpodError(f"Unexpected pod list response: {pods!r}")
    return pods


def get_pod(api_key: str, api_url: str, pod_id: str) -> dict[str, Any]:
    pod = api_request(
        api_key,
        api_url,
        "GET",
        f"/pods/{pod_id}",
        params={"includeMachine": "true"},
    )
    if not isinstance(pod, dict):
        raise RunpodError(f"Unexpected pod response for {pod_id}: {pod!r}")
    return pod


def create_pod(api_key: str, api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    pod = api_request(api_key, api_url, "POST", "/pods", payload=payload)
    if not isinstance(pod, dict):
        raise RunpodError(f"Unexpected pod creation response: {pod!r}")
    return pod


def create_pod_with_retry(
    api_key: str,
    api_url: str,
    payload: dict[str, Any],
    *,
    retries: int,
    initial_delay: float,
    max_delay: float,
    retry_with_any_cuda: bool,
) -> dict[str, Any]:
    max_attempts = retries + 1
    attempt = 1
    active_payload = dict(payload)
    relaxed_cuda = False
    while True:
        try:
            return create_pod(api_key, api_url, active_payload)
        except Exception as exc:
            if (
                is_capacity_error(exc)
                and retry_with_any_cuda
                and not relaxed_cuda
                and active_payload.get("allowedCudaVersions")
            ):
                previous = ", ".join(active_payload["allowedCudaVersions"])
                active_payload = dict(active_payload)
                active_payload.pop("allowedCudaVersions", None)
                relaxed_cuda = True
                eprint(
                    "Create attempt "
                    f"{attempt}/{max_attempts} hit a capacity error. "
                    f"Retrying with any CUDA version instead of {previous}."
                )
                continue

            if not is_capacity_error(exc) or attempt >= max_attempts:
                raise format_create_pod_error(
                    active_payload,
                    exc,
                    attempts=attempt,
                    max_attempts=max_attempts,
                ) from exc

            delay = min(initial_delay * (1.5 ** (attempt - 1)), max_delay)
            request_summary = format_capacity_request(active_payload)
            assert isinstance(exc, RunpodApiError)
            eprint(
                "Create attempt "
                f"{attempt}/{max_attempts} hit a capacity error for {request_summary}. "
                f"API said: {exc.error_text}. Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
            attempt += 1


def delete_pod(api_key: str, api_url: str, pod_id: str) -> None:
    api_request(api_key, api_url, "DELETE", f"/pods/{pod_id}")


def get_pod_if_exists(
    api_key: str,
    api_url: str,
    pod_id: str,
) -> dict[str, Any] | None:
    try:
        return get_pod(api_key, api_url, pod_id)
    except RunpodApiError as exc:
        if exc.status == 404:
            return None
        raise


def parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise RunpodError(f"Invalid env assignment {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise RunpodError(f"Invalid env assignment {item!r}; key is empty")
        env[key] = value
    return env


def parse_inline_env_command(
    command: list[str],
) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    index = 0
    while index < len(command) and INLINE_ENV_RE.match(command[index]):
        key, value = command[index].split("=", 1)
        env[key] = value
        index += 1
    return env, command[index:]


def looks_like_shell_command(command: str) -> bool:
    if any(char.isspace() for char in command):
        return True
    return any(char in SHELL_META_CHARS for char in command)


def describe_command(command: list[str]) -> str:
    if not command:
        return "<interactive shell>"
    if len(command) == 1:
        return command[0]
    return shlex.join(command)


def build_remote_exec_command(remote_dir: str, command: list[str]) -> str:
    mode = "login"
    payload: object = ""
    env: dict[str, str] = {}

    if command:
        if len(command) == 1 and looks_like_shell_command(command[0]):
            mode = "shell"
            payload = command[0]
        else:
            env, argv = parse_inline_env_command(command)
            if not argv:
                raise RunpodError(
                    "Command contains environment assignments but no executable"
                )
            mode = "exec"
            payload = argv

    remote_argv = [
        "python3",
        "-c",
        REMOTE_EXEC_PY,
        remote_dir,
        mode,
        json.dumps(payload),
        json.dumps(env),
    ]
    return shlex.join(remote_argv)


def image_push_ref(image_name: str) -> str:
    if "@" in image_name:
        raise RunpodError("Image rebuild does not support digest references; use a tag")
    last_segment = image_name.rsplit("/", 1)[-1]
    if ":" not in last_segment:
        return f"{image_name}:latest"
    return image_name


def rebuild_image(repo_root: Path, image_name: str, *, dry_run: bool) -> None:
    push_ref = image_push_ref(image_name)
    build_command = ["docker", "build", "-t", push_ref, str(repo_root)]
    push_command = ["docker", "push", push_ref]

    eprint("Rebuild image commands:")
    eprint("  " + shlex.join(build_command))
    eprint("  " + shlex.join(push_command))

    if dry_run:
        return

    run_subprocess(build_command)
    run_subprocess(push_command)


def ssh_transport(host: str, port: int) -> list[str]:
    return [
        "ssh",
        *SSH_OPTIONS,
        "-p",
        str(port),
        f"root@{host}",
    ]


def ssh_shell(host: str, port: int, command: str) -> list[str]:
    return [*ssh_transport(host, port), command]


def run_subprocess(
    command: list[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, timeout=timeout)


def port_mapping(pod: dict[str, Any], port: str) -> int | None:
    mappings = pod.get("portMappings") or {}
    if not isinstance(mappings, dict):
        return None
    value = mappings.get(port) or mappings.get(str(port))
    if value is None:
        return None
    return int(value)


def pod_data_center(pod: dict[str, Any]) -> str | None:
    machine = pod.get("machine") or {}
    if isinstance(machine, dict):
        for key in ("dataCenterId", "dataCenter", "podHostId"):
            value = machine.get(key)
            if isinstance(value, str) and value:
                return value
    value = pod.get("dataCenterId")
    if isinstance(value, str) and value:
        return value
    return None


def format_pod_summary(pod: dict[str, Any]) -> str:
    gpu = pod.get("gpu") or {}
    display_name = (
        gpu.get("displayName")
        or pod.get("machine", {}).get("gpuDisplayName")
        or "unknown"
    )
    public_ip = pod.get("publicIp") or "-"
    ssh_port = port_mapping(pod, "22")
    ssh_text = str(ssh_port) if ssh_port is not None else "-"
    cost = pod.get("costPerHr") or pod.get("adjustedCostPerHr") or "-"
    data_center = pod_data_center(pod) or "-"
    return (
        f"id={pod.get('id')} name={pod.get('name')} status={pod.get('desiredStatus')} "
        f"gpu={display_name} dc={data_center} cost/hr={cost} ip={public_ip} ssh={ssh_text}"
    )


def print_pods(
    pods: list[dict[str, Any]], managed_name: str, *, show_all: bool
) -> None:
    visible = list(pods)
    if not show_all:
        visible = [pod for pod in visible if pod.get("desiredStatus") != "TERMINATED"]
    if not visible:
        eprint("No pods found.")
        return
    for pod in sorted(
        visible, key=lambda item: (item.get("name") or "", item.get("id") or "")
    ):
        marker = "*" if pod.get("name") == managed_name else " "
        eprint(f"{marker} {format_pod_summary(pod)}")


def cleanup_managed_pods(
    api_key: str,
    api_url: str,
    managed_name: str,
    *,
    repo_root: Path | None = None,
) -> int:
    pods = get_pods(api_key, api_url)
    candidates = [
        pod
        for pod in pods
        if pod.get("name") == managed_name and pod.get("desiredStatus") != "TERMINATED"
    ]
    if not candidates:
        eprint(f"No matching active pods found for name={managed_name}")
        return 0

    stale_pods: list[dict[str, Any]] = []
    for pod in candidates:
        if repo_root is None:
            stale_pods.append(pod)
            continue

        pod_id = str(pod["id"])
        lock_path = pod_lock_path(repo_root, pod_id)
        if not lock_path.exists():
            stale_pods.append(pod)
            continue

        probe_lock = FileLock(str(lock_path))
        try:
            probe_lock.acquire(blocking=False)
        except FileLockTimeout:
            eprint(
                f"Skipping active same-name pod with live local lock: {format_pod_summary(pod)}",
            )
            continue
        except OSError:
            stale_pods.append(pod)
            continue
        else:
            stale_pods.append(pod)
        finally:
            with suppress(Exception):
                probe_lock.release()

        with suppress(OSError):
            lock_path.unlink()

    if not stale_pods:
        eprint(f"No matching active pods found for name={managed_name}")
        return 0

    for pod in stale_pods:
        eprint(f"Deleting pod: {format_pod_summary(pod)}")
        delete_pod(api_key, api_url, str(pod["id"]))
    if repo_root is not None:
        clear_managed_pod_state(repo_root, managed_name)
    return len(stale_pods)


def find_reusable_managed_pod(
    api_key: str,
    api_url: str,
    managed_name: str,
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    state = read_managed_pod_state(repo_root, managed_name)
    if state is None:
        return None
    pod_id = state.get("pod_id")
    if not isinstance(pod_id, str) or not pod_id:
        clear_managed_pod_state(repo_root, managed_name)
        return None

    pod = get_pod_if_exists(api_key, api_url, pod_id)
    if pod is None or pod.get("desiredStatus") == "TERMINATED":
        clear_managed_pod_state(repo_root, managed_name)
        return None
    if pod.get("name") != managed_name:
        clear_managed_pod_state(repo_root, managed_name)
        return None
    return pod


def warn_other_active_pods(
    api_key: str,
    api_url: str,
    managed_name: str,
    *,
    context: str,
    exclude_pod_id: str | None = None,
) -> None:
    try:
        pods = get_pods(api_key, api_url)
    except Exception as exc:  # pragma: no cover - best effort warning
        eprint(f"Warning: failed to list other active pods at {context}: {exc}")
        return

    others = []
    for pod in pods:
        if pod.get("desiredStatus") == "TERMINATED":
            continue
        if exclude_pod_id is not None and str(pod.get("id")) == exclude_pod_id:
            continue
        if pod.get("name") == managed_name:
            continue
        others.append(pod)

    if not others:
        return

    eprint(f"Warning: found {len(others)} other active pod(s) at {context}:")
    for pod in sorted(
        others, key=lambda item: (item.get("name") or "", item.get("id") or "")
    ):
        eprint(f"  {format_pod_summary(pod)}")


def progress_line(elapsed: float, pod: dict[str, Any], ssh_ready: bool) -> str:
    ip = pod.get("publicIp") or "-"
    ssh_port = port_mapping(pod, "22")
    ssh_text = str(ssh_port) if ssh_port is not None else "-"
    if ssh_ready or (ip != "-" and ssh_port is not None):
        wait_text = "Waiting for ssh"
    else:
        wait_text = "Waiting for network (this might take up to 10 minutes)"
    return (
        f"[{int(elapsed):4d}s] {wait_text} "
        f"status={pod.get('desiredStatus', '?'):>10} ip={ip:<15} ssh={ssh_text:<6}"
    )


def is_terminal_status(status: object) -> bool:
    return status in {"EXITED", "TERMINATED"}


def wait_for_ssh(
    api_key: str,
    api_url: str,
    pod_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    last_line = ""
    last_status = None
    announced_network = False
    while True:
        pod = get_pod(api_key, api_url, pod_id)
        elapsed = time.monotonic() - start
        status = pod.get("desiredStatus")
        ip = pod.get("publicIp")
        ssh_port = port_mapping(pod, "22")
        ssh_ready = False

        if status != last_status:
            status_change = pod.get("lastStatusChange") or "unknown"
            if last_line:
                eprint()
                last_line = ""
            eprint(f"Status change: {status} ({status_change})")
            last_status = status
        if ip and ssh_port and not announced_network:
            if last_line:
                eprint()
                last_line = ""
            eprint(f"Network ready: ip={ip} ssh={ssh_port}")
            announced_network = True

        if ip and ssh_port:
            try:
                run_subprocess(
                    [*ssh_transport(str(ip), ssh_port), "true"],
                    timeout=10,
                )
                ssh_ready = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                ssh_ready = False

        line = progress_line(elapsed, pod, ssh_ready)
        if line != last_line:
            eprint(f"\r{line}", end="", flush=True)
            last_line = line

        if ssh_ready:
            if last_line:
                eprint()
            eprint(f"Pod is ready: {format_pod_summary(pod)}")
            return pod, elapsed

        if is_terminal_status(pod.get("desiredStatus")):
            if last_line:
                eprint()
            status_change = pod.get("lastStatusChange") or "unknown reason"
            raise RunpodError(
                f"Pod {pod_id} entered terminal state {pod.get('desiredStatus')} before becoming ready: {status_change}"
            )

        if elapsed >= timeout:
            if last_line:
                eprint()
            raise RunpodError(
                f"Timed out waiting for pod {pod_id} to become reachable over SSH"
            )

        time.sleep(poll_interval)


def ensure_remote_dir(ip: str, ssh_port: int, remote_dir: str) -> None:
    run_subprocess(
        [*ssh_transport(ip, ssh_port), "mkdir", "-p", remote_dir],
    )


def rsync_repo(ip: str, ssh_port: int, repo_root: Path, remote_dir: str) -> None:
    if shutil.which("rsync") is None:
        raise RunpodError("rsync is not installed locally")

    ensure_remote_dir(ip, ssh_port, remote_dir)
    ssh_cmd = " ".join(
        shlex.quote(part) for part in ["ssh", *SSH_OPTIONS, "-p", str(ssh_port)]
    )
    command = [
        "rsync",
        "-az",
        "--delete",
        "--no-owner",
        "--no-group",
    ]
    for pattern in RSYNC_EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend(
        [
            "-e",
            ssh_cmd,
            f"{repo_root}/",
            f"root@{ip}:{remote_dir}/",
        ]
    )
    eprint(f"Syncing repo to {ip}:{remote_dir}")
    run_subprocess(command)


def run_remote_command(
    ip: str, ssh_port: int, remote_dir: str, command: list[str]
) -> int:
    remote = build_remote_exec_command(remote_dir, command)
    completed = subprocess.run(
        ssh_shell(ip, ssh_port, remote),
        text=True,
        check=False,
    )
    return completed.returncode


def build_payload(args: argparse.Namespace, pod_name: str) -> dict[str, Any]:
    user_env = parse_env(args.env)
    payload: dict[str, Any] = {
        "allowedCudaVersions": args.allowed_cuda_versions,
        "computeType": "GPU",
        "gpuCount": args.gpu_count,
        "gpuTypeIds": args.gpu_type_ids,
        "imageName": args.image_name,
        "interruptible": args.interruptible,
        "name": args.name or pod_name,
        "ports": args.ports,
        "supportPublicIp": args.support_public_ip,
        "env": user_env,
    }
    if args.cloud_type:
        payload["cloudType"] = args.cloud_type
    if args.data_center_ids:
        payload["dataCenterIds"] = args.data_center_ids
        payload["dataCenterPriority"] = args.data_center_priority
    if args.container_disk_gb is not None:
        payload["containerDiskInGb"] = args.container_disk_gb
    if args.volume_gb is not None:
        payload["volumeInGb"] = args.volume_gb
        payload["volumeMountPath"] = args.volume_mount_path
    return payload


def redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command on a Runpod pod after syncing the local repo.",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--api-url")
    parser.add_argument("--name", help="Override the generated pod name")
    parser.add_argument(
        "--data-center-id",
        dest="data_center_ids",
        action="append",
        help="Preferred Runpod data center ID. Repeatable.",
    )
    parser.add_argument(
        "--any-data-center",
        action="store_true",
        help="Do not apply the default data center allowlist.",
    )
    parser.add_argument(
        "--data-center-priority",
        choices=["availability", "price"],
        default=DEFAULT_DATA_CENTER_PRIORITY,
        help="Priority rule when multiple --data-center-id values are supplied.",
    )
    parser.add_argument(
        "--allowed-cuda-version",
        dest="allowed_cuda_versions",
        action="append",
        help="Allowed CUDA version. Repeatable. Defaults to 13.0.",
    )
    parser.add_argument(
        "--gpu-type-id",
        dest="gpu_type_ids",
        action="append",
        help="GPU type ID. Repeatable. Defaults to NVIDIA B200.",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument(
        "--interruptible", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--port",
        dest="ports",
        action="append",
        help="Exposed port specification like 22/tcp. Repeatable.",
    )
    parser.add_argument(
        "--support-public-ip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the pod. Repeatable.",
    )
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--volume-gb", type=int)
    parser.add_argument("--volume-mount-path", default="/workspace")
    parser.add_argument(
        "--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument(
        "--create-retries",
        type=int,
        default=DEFAULT_CREATE_RETRIES,
        help="Retry pod creation on transient capacity errors this many times.",
    )
    parser.add_argument(
        "--create-retry-initial-delay",
        type=float,
        default=DEFAULT_CREATE_RETRY_INITIAL_DELAY,
        help="Initial delay in seconds before retrying a capacity-related create failure.",
    )
    parser.add_argument(
        "--create-retry-max-delay",
        type=float,
        default=DEFAULT_CREATE_RETRY_MAX_DELAY,
        help="Maximum backoff delay in seconds for capacity-related create retries.",
    )
    parser.add_argument(
        "--retry-with-any-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On capacity errors, retry once without an allowed CUDA version constraint.",
    )
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        help="Do not delete the pod after the command exits.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start or reuse a persistent pod and leave it running for later requests.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete the current matching pod, start a fresh persistent pod, and leave it running.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the create-pod request and exit."
    )
    parser.add_argument(
        "--rebuild-image",
        action="store_true",
        help="Build the local Docker image with --image-name and push it to the registry.",
    )
    parser.add_argument("--list", action="store_true", help="List pods and exit.")
    parser.add_argument(
        "--cleanup",
        "--stop",
        action="store_true",
        help="Delete matching running pods for the generated pod name and exit.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute on the pod. Omit for an interactive login shell.",
    )
    args = parser.parse_args()

    if not args.allowed_cuda_versions:
        args.allowed_cuda_versions = list(DEFAULT_ALLOWED_CUDA_VERSIONS)
    if args.any_data_center and args.data_center_ids:
        parser.error("Do not combine --any-data-center with --data-center-id")
    if args.any_data_center:
        args.data_center_ids = []
    elif args.data_center_ids is None:
        args.data_center_ids = list(DEFAULT_DATA_CENTER_IDS)
    if not args.gpu_type_ids:
        args.gpu_type_ids = list(DEFAULT_GPU_TYPE_IDS)
    if not args.ports:
        args.ports = list(DEFAULT_PORTS)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    actions = [args.list, args.cleanup, args.rebuild_image]
    if sum(bool(action) for action in actions) > 1:
        parser.error("--list, --cleanup, and --rebuild-image are mutually exclusive")
    if (args.list or args.cleanup or args.rebuild_image) and args.command:
        parser.error("Do not pass a command with --list, --cleanup, or --rebuild-image")
    if args.restart and (args.list or args.cleanup or args.rebuild_image):
        parser.error(
            "--restart cannot be combined with --list, --cleanup, or --rebuild-image"
        )
    if (
        not args.list
        and not args.cleanup
        and not args.rebuild_image
        and repo_root.name != "helion"
    ):
        parser.error(f"Expected to run from the Helion repo, found {repo_root}")
    if args.create_retries < 0:
        parser.error("--create-retries must be non-negative")
    if args.create_retry_initial_delay <= 0:
        parser.error("--create-retry-initial-delay must be positive")
    if args.create_retry_max_delay <= 0:
        parser.error("--create-retry-max-delay must be positive")

    return args


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)

    if args.rebuild_image:
        rebuild_image(repo_root, args.image_name, dry_run=args.dry_run)
        return 0

    api_key, api_url = load_api_settings(args)
    pod_name = compute_pod_name(repo_root)
    managed_name = args.name or pod_name

    if args.list:
        pods = get_pods(api_key, api_url)
        print_pods(pods, managed_name, show_all=True)
        return 0

    if args.cleanup:
        cleanup_managed_pods(api_key, api_url, managed_name, repo_root=repo_root)
        return 0

    payload = build_payload(args, pod_name)

    if args.dry_run:
        if args.restart:
            eprint(f"Would restart matching pod(s) for name={managed_name}")
        eprint(json.dumps(redacted_payload(payload), indent=2, sort_keys=True))
        return 0

    if args.restart:
        cleanup_managed_pods(api_key, api_url, managed_name, repo_root=repo_root)
        args.start = True

    eprint(f"Repo root: {repo_root}")
    eprint(f"Pod name: {payload['name']}")
    eprint(f"Image: {payload['imageName']}")
    eprint(f"GPU types: {', '.join(payload['gpuTypeIds'])}")
    eprint(f"Allowed CUDA versions: {', '.join(payload['allowedCudaVersions'])}")
    if args.data_center_ids:
        eprint(
            "Data centers: "
            f"{', '.join(args.data_center_ids)} "
            f"(priority={args.data_center_priority})"
        )
    else:
        eprint("Data centers: any")
    eprint(
        "Create retries: "
        f"{args.create_retries} "
        f"(initial={args.create_retry_initial_delay:.0f}s max={args.create_retry_max_delay:.0f}s "
        f"retry_with_any_cuda={args.retry_with_any_cuda})"
    )
    warn_other_active_pods(api_key, api_url, str(payload["name"]), context="start")

    pod_id: str | None = None
    pod_lock: PodLock | None = None
    created_persistent_pod = False
    reused_managed_pod = False
    overall_start = time.monotonic()
    try:
        pod = find_reusable_managed_pod(
            api_key,
            api_url,
            str(payload["name"]),
            repo_root=repo_root,
        )
        if pod is not None:
            reused_managed_pod = True
            pod_id = str(pod["id"])
            pod_lock = PodLock(repo_root, pod_id)
            pod_lock.acquire()
            eprint(f"Reusing persistent pod: {format_pod_summary(pod)}")
        else:
            pod = create_pod_with_retry(
                api_key,
                api_url,
                payload,
                retries=args.create_retries,
                initial_delay=args.create_retry_initial_delay,
                max_delay=args.create_retry_max_delay,
                retry_with_any_cuda=args.retry_with_any_cuda,
            )
            pod_id = str(pod["id"])
            pod_lock = PodLock(repo_root, pod_id)
            pod_lock.acquire()
            eprint(f"Created pod: {format_pod_summary(pod)}")
            if args.start:
                write_managed_pod_state(repo_root, str(payload["name"]), pod_id)
                created_persistent_pod = True

        pod, startup_elapsed = wait_for_ssh(
            api_key,
            api_url,
            pod_id,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
        ip = str(pod["publicIp"])
        ssh_port = port_mapping(pod, "22")
        if ssh_port is None:
            raise RunpodError(f"Pod {pod_id} does not expose TCP port 22")

        sync_start = time.monotonic()
        rsync_repo(ip, ssh_port, repo_root, args.remote_dir)
        sync_elapsed = time.monotonic() - sync_start

        if args.start and not args.command:
            total_elapsed = time.monotonic() - overall_start
            eprint(
                "Timing summary: "
                f"startup={startup_elapsed:.1f}s "
                f"sync={sync_elapsed:.1f}s "
                f"total={total_elapsed:.1f}s"
            )
            return 0

        eprint(f"Running remote command: {describe_command(args.command)}")
        command_start = time.monotonic()
        returncode = run_remote_command(ip, ssh_port, args.remote_dir, args.command)
        command_elapsed = time.monotonic() - command_start
        total_elapsed = time.monotonic() - overall_start
        eprint(
            "Timing summary: "
            f"startup={startup_elapsed:.1f}s "
            f"sync={sync_elapsed:.1f}s "
            f"command={command_elapsed:.1f}s "
            f"total={total_elapsed:.1f}s"
        )
        return returncode
    finally:
        leave_running = args.keep_pod or args.start or reused_managed_pod
        if pod_id and not leave_running:
            try:
                eprint(f"Deleting pod {pod_id}")
                delete_pod(api_key, api_url, pod_id)
            except Exception as exc:  # pragma: no cover - cleanup best effort
                eprint(f"Warning: failed to delete pod {pod_id}: {exc}")
        elif pod_id and (created_persistent_pod or reused_managed_pod):
            eprint(
                "Warning: pod is left running. Call --cleanup to shut it down.",
            )
        if pod_lock is not None:
            pod_lock.release()
        warn_other_active_pods(
            api_key, api_url, str(payload["name"]), context="end", exclude_pod_id=pod_id
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        eprint("\nInterrupted")
        raise SystemExit(130) from exc
    except RunpodError as exc:
        eprint(f"Error: {exc}")
        raise SystemExit(1) from exc

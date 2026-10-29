"""Coding tools for Code mode.

  run_python:    run a Python snippet in a separate process (opt-in, see CODE_EXECUTION in .env)
  package_info:  latest version, release date and metadata from PyPI / npm (live, no key)
  github_file:   read a file or list a directory from a public GitHub repository
"""
import os
import subprocess
import sys
import tempfile
from datetime import datetime

from langchain_core.tools import tool

from chatbot.config import settings
from chatbot.tools.common import cached, http_get, safe, to_text, trim

RUN_TIMEOUT_S = 20
# Only what Python needs to start; API keys and other secrets from .env are NOT passed to user code
_SAFE_ENV_VARS = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT", "COMSPEC", "LANG", "HOME", "USERPROFILE")


def execute_python(code: str, timeout: int = RUN_TIMEOUT_S) -> dict:
    """Run code in a fresh, isolated Python process in an empty temp folder. Not a security sandbox:
    it runs on this machine with the server's permissions, which is why it's off on public deployments."""
    with tempfile.TemporaryDirectory(prefix="pulse_run_") as work:
        script = os.path.join(work, "snippet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        env = {k: os.environ[k] for k in _SAFE_ENV_VARS if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"  # matplotlib without a display
        started = datetime.now()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script],  # -I: ignore env vars/user site, don't import from cwd
                cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
            out = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as e:
            out = {"exit_code": None, "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                   "stderr": f"Timed out after {timeout}s (infinite loop or waiting for input?)"}
        out["duration_ms"] = int((datetime.now() - started).total_seconds() * 1000)
        out["stdout"] = trim(out["stdout"], 6000)
        out["stderr"] = trim(out["stderr"], 3000)
        return out


@tool
@safe
def run_python(code: str) -> str:
    """Execute a Python 3 snippet and return stdout, stderr and the exit code. Use it to test code you
    wrote, verify outputs, reproduce a bug, or compute something precisely. The snippet runs in a fresh
    process (no state is kept between calls, no input() available, 20s limit). Print what you want to see.
    Installed libraries include pandas, numpy, httpx, bs4; no internet guarantees."""
    if not settings.code_execution:
        return "Error: code execution is disabled on this server (set CODE_EXECUTION=on in .env to enable)."
    return to_text(execute_python(code))


@tool
@safe
@cached(ttl=900)
def package_info(name: str, ecosystem: str = "pypi") -> str:
    """Live package metadata: latest version, release date, Python/Node requirements, description,
    links and recent versions. ecosystem: pypi (Python) or npm (JavaScript/TypeScript).
    Use before recommending a library version or install command."""
    eco = ecosystem.lower().strip()
    if eco in ("pypi", "pip", "python"):
        data = http_get(f"https://pypi.org/pypi/{name.strip()}/json").json()
        info = data["info"]
        releases = sorted(
            ((v, files[0]["upload_time"][:10]) for v, files in data.get("releases", {}).items() if files),
            key=lambda x: x[1], reverse=True,
        )
        latest_date = next((d for v, d in releases if v == info["version"]), None)
        return to_text(
            {
                "source": "PyPI", "package": info["name"], "latest_version": info["version"],
                "released": latest_date, "summary": info.get("summary"),
                "requires_python": info.get("requires_python"), "license": info.get("license_expression") or (info.get("license") or "")[:60],
                "homepage": info.get("home_page") or (info.get("project_urls") or {}).get("Homepage"),
                "docs": (info.get("project_urls") or {}).get("Documentation"),
                "pypi_url": f"https://pypi.org/project/{info['name']}/",
                "install": f"pip install {info['name']}=={info['version']}",
                "recent_versions": [f"{v} ({d})" for v, d in releases[:8]],
            }
        )
    if eco in ("npm", "node", "js", "javascript", "typescript"):
        data = http_get(f"https://registry.npmjs.org/{name.strip()}").json()
        latest = data.get("dist-tags", {}).get("latest")
        times = {k: v for k, v in data.get("time", {}).items() if k not in ("created", "modified")}
        recent = sorted(times.items(), key=lambda x: x[1], reverse=True)[:8]
        ver = data.get("versions", {}).get(latest, {})
        return to_text(
            {
                "source": "npm", "package": data.get("name"), "latest_version": latest,
                "released": (times.get(latest) or "")[:10], "description": data.get("description"),
                "node_engines": ver.get("engines"), "license": ver.get("license"),
                "homepage": data.get("homepage"), "npm_url": f"https://www.npmjs.com/package/{data.get('name')}",
                "install": f"npm install {data.get('name')}@{latest}",
                "dist_tags": data.get("dist-tags"),
                "recent_versions": [f"{v} ({t[:10]})" for v, t in recent],
            }
        )
    return "Error: ecosystem must be 'pypi' or 'npm'."


@tool
@safe
@cached(ttl=600)
def github_file(repo: str, path: str = "", ref: str = "") -> str:
    """Read source code from a public GitHub repository. repo: 'owner/name' (or a github.com URL).
    path: a file path to read (e.g. 'src/app.py'), or a folder / '' to list its contents.
    Start with path='' to see the layout, then README.md, then specific files."""
    repo = repo.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    repo = "/".join(repo.split("/")[:2])
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    params = {"ref": ref} if ref else None
    data = http_get(f"https://api.github.com/repos/{repo}/contents/{path.strip('/')}", headers=headers, params=params).json()
    if isinstance(data, list):  # a directory
        entries = sorted(data, key=lambda e: (e.get("type") != "dir", e.get("name", "").lower()))
        return to_text(
            {
                "repo": repo, "path": path or "/", "url": f"https://github.com/{repo}/tree/{ref or 'HEAD'}/{path}".rstrip("/"),
                "entries": [f"{'📁' if e['type'] == 'dir' else '📄'} {e['name']}" + (f"  ({e['size']} B)" if e["type"] == "file" else "") for e in entries[:200]],
            }
        )
    if data.get("type") != "file":
        return f"Error: {repo}/{path} is not a file or folder."
    raw = http_get(data["download_url"]).text
    return f"FILE {repo}/{data['path']} ({data.get('size')} bytes)\nURL: {data.get('html_url')}\n\n" + trim(raw, 8000)

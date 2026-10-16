# [MODULE DOC]
# HELPER: data layer for the Harbor UI: reads inside task directories (task.toml,
# instruction.md, file tree) to build task list rows and task detail pages, with a
# process-wide per-dataset row cache; serves single task files safely; decides whether a
# task would read the server's environment or host files (task.toml and compose-file
# analysis); searches Harbor datasets and Inference Providers models on the Hub; and adds
# or removes Hub datasets from the page (background job: layout/size check, bucket copy or
# download, symlink refusal, indexing). Never runs a sandbox or model.
"""What the Harbor UI shows about a dataset: its tasks, what is in each one, and what a task runs with.

Discovery (`tasks.py`) lists task directories and deliberately reads nothing inside them, because it
is on the latency path of every Task API call. A person choosing a task needs more than a directory
name: a title to find it by, the files to judge it by, and the settings it will run with. So this
module does read inside task directories, but only for the UI, on demand, and with a per-dataset
cache so a page load never walks a dataset twice.

Nothing here boots a sandbox or calls a model.
"""

from __future__ import annotations

import os
import posixpath
import re
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .tasks import HarborTaskProvider, own_file as _own, resolve_task_dirs, task_root

# Metadata keys that hold the answer. Harbor puts no rules on `[metadata]`, and some datasets keep
# the gold answer there (`gold_answer = "4"`). The raw `task.toml` stays viewable, the dataset is
# public anyway, but a summary someone skims before running a model should not hand them the answer.
_ANSWER_KEY = re.compile(r"(gold|answer|solution|expected|reference_output)", re.I)

# The file tree is for reading a task, not for mirroring a repository: a task with a vendored
# codebase can hold tens of thousands of files, and the browser only needs the shape.
MAX_TREE_FILES = 2000
MAX_FILE_BYTES = 256 * 1024

_ROWS: dict[str, list[dict[str, Any]]] = {}
_ROWS_INFLIGHT: dict[str, object] = {}
_ROWS_LOCK = threading.Lock()


def _hub_not_found(exc: Exception) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the exception is FileNotFoundError or carries an HTTP 404 response.
    """Whether a Hub exception specifically says the requested entry does not exist."""
    if isinstance(exc, FileNotFoundError):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def _first_link(root: Path) -> Path | None:
    # [FUNCTION DOC]
    # HELPER: walks root without following links and returns the first symlink found (file or folder), or None.
    """A symbolic link under `root`, file or folder, without following any; `None` if there is none."""
    folders = [root]
    while folders:
        with os.scandir(folders.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    return Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    folders.append(Path(entry.path))
    return None


def _links_a_task(dataset: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if dataset/tasks is a symlink or any direct entry in it is. Raises OSError
    # if the tasks folder cannot be listed.
    """Whether `dataset/tasks`, or a task folder in it, is a link: what discovery would follow.

    One directory listing, so it is cheap enough to ask of every added dataset at startup; links
    further down are never followed by the page, the file viewer or `reads_environment`.
    """
    tasks = dataset / "tasks"
    if tasks.is_symlink():
        return True
    with os.scandir(tasks) as entries:
        return any(entry.is_symlink() for entry in entries)


def _toml(task_dir: Path | None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: parses the task's task.toml (via the containment-checked own_file); returns {}
    # on missing/unreadable/invalid files or a None task dir.
    try:
        return tomllib.loads(_own(task_dir, "task.toml").read_text(errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _first_line(task_dir: Path | None, limit: int = 160) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the first non-empty line of instruction.md with leading "#" stripped,
    # truncated to limit; "" if unavailable.
    """The first meaningful line of the instruction, used as a title when a task declares none."""
    try:
        with _own(task_dir, "instruction.md").open(errors="replace") as fh:
            for line in fh:
                text = line.strip().lstrip("#").strip()
                if text:
                    return text[:limit]
    except OSError:
        pass
    return ""


_MD = re.compile(
    r"`+|\*\*|__|^\s*[-*+>]\s+|^\s*\d+[.)]\s+|!?\[([^\]]*)\]\([^)]*\)", re.M
)


def _paragraphs(task_dir: Path | None, limit: int = 3, chars: int = 360) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: reads the first 6000 chars of instruction.md and returns up to limit prose
    # paragraphs (skipping headings and fenced code), with basic markdown syntax stripped and
    # each truncated to chars. Empty list if unreadable.
    """The instruction's first few prose paragraphs as plain text, headings and code left out.

    Only the start of the file is read: a brief is two lines on a card.
    """
    try:
        with _own(task_dir, "instruction.md").open(errors="replace") as fh:
            head = fh.read(6000)
    except OSError:
        return []
    out, block, fenced = [], [], False
    for line in head.splitlines() + [""]:
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith("#"):
            continue
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            text = _MD.sub(lambda m: m.group(1) or "", " ".join(block)).strip()
            if text:
                out.append(text[:chars])
            block = []
            if len(out) >= limit:
                break
    return out


def _strings(value: Any) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: returns [value] for a string, stringified str/int/float items for a list/tuple, else [].
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if isinstance(v, (str, int, float))]
    return []


def task_row(index: int, task_dir: Path, spec: str | None = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Builds a task list row: index, directory name, title (metadata title, task description,
    # first instruction line, or directory name; max 240 chars), category (category/domain),
    # difficulty (difficulty/difficulty_tier), up to 8 deduplicated keywords from [task] and
    # [metadata] (excluding the category), and instruction paragraphs (used for briefs and
    # removed later). Reads only within the dataset (task_root).
    """One task as the task list shows it.

    Harbor's schema fixes `[task]` (name, description, keywords) but leaves `[metadata]` free-form,
    so a title, a category and a difficulty are each looked for under the names datasets actually
    use, falling back to the instruction's first line for the title.

    Args:
        index (`int`):
            The task's position in its dataset, which is its identity everywhere downstream.
        task_dir (`Path`):
            The task directory.
        spec (`str`, *optional*):
            Its dataset, which every file read must stay inside (`task_root`).

    Returns:
        `dict` with `index`, `name`, `title`, `category`, `difficulty` and `keywords`.
    """
    root = task_root(spec, task_dir)
    doc = _toml(root)
    task, meta = doc.get("task") or {}, doc.get("metadata") or {}
    title = (
        meta.get("title")
        or task.get("description")
        or _first_line(root)
        or task_dir.name
    )
    keywords: list[str] = []
    for k in _strings(task.get("keywords")) + _strings(meta.get("keywords")):
        if k not in keywords and k != meta.get("category"):
            keywords.append(k)
    return {
        "index": index,
        "name": task_dir.name,
        "title": str(title)[:240],
        "category": str(meta.get("category") or meta.get("domain") or ""),
        "difficulty": str(meta.get("difficulty") or meta.get("difficulty_tier") or ""),
        "keywords": keywords[:8],
        "paragraphs": _paragraphs(root),
    }


def _briefs(rows: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC]
    # HELPER: in place, replaces each row's paragraphs with a "brief": its first paragraph
    # (max 220 chars) that is not shared by at least max(3, n/5) rows and does not repeat the title.
    """Give each row a `brief`: its first paragraph that says something about this task.

    Many datasets open every instruction with the same preamble ("You are an agent, your working
    directory is /app"), which on a card reads as the same sentence 2,000 times. A paragraph shared by
    more than a fifth of a dataset's tasks is that preamble, so it is skipped, as is one that only
    repeats the title.
    """
    counts: dict[str, int] = {}
    for r in rows:
        for p in set(r.get("paragraphs") or []):
            counts[p] = counts.get(p, 0) + 1
    common = max(3, len(rows) // 5)
    for r in rows:
        title = r["title"].strip().rstrip(".").lower()
        paras = r.pop("paragraphs", None) or []
        r["brief"] = next(
            (
                p[:220]
                for p in paras
                if counts.get(p, 0) < common
                and p.strip().rstrip(".").lower() != title
                and not title.startswith(p.strip().rstrip(".").lower()[:60])
            ),
            "",
        )


def task_rows(spec: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Returns cached rows for a dataset, or resolves its task dirs and builds rows in a thread
    # pool (OPENENV_UI_INDEX_WORKERS, default 16), adds briefs, and caches them unless the
    # dataset was invalidated (removed) during indexing. Errors propagate and clear the in-flight marker.
    """Every task in a dataset as the task list shows it, cached per dataset.

    Reading one small `task.toml` per task is milliseconds on local disk but far slower on a
    mounted bucket, so the reads run in parallel and the result is kept for the process lifetime.

    Args:
        spec (`str`):
            Dataset spec: HF dataset repo, local directory, or Harbor registry `name@version`.

    Returns:
        `list[dict]`, one row per task, in index order.
    """
    with _ROWS_LOCK:
        if spec in _ROWS:
            return _ROWS[spec]
        token = object()
        _ROWS_INFLIGHT[spec] = token
    try:
        dirs = resolve_task_dirs(spec)
        workers = int(os.environ.get("OPENENV_UI_INDEX_WORKERS", "16"))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows = list(
                pool.map(lambda pair: task_row(*pair, spec=spec), enumerate(dirs))
            )
        _briefs(rows)
    except BaseException:
        with _ROWS_LOCK:
            if _ROWS_INFLIGHT.get(spec) is token:
                _ROWS_INFLIGHT.pop(spec, None)
        raise
    with _ROWS_LOCK:
        # Removal invalidates both caches while indexing can still be reading files. Do not let that
        # in-flight result resurrect rows for a dataset whose backing folder has been deleted.
        if _ROWS_INFLIGHT.get(spec) is token:
            _ROWS[spec] = rows
            _ROWS_INFLIGHT.pop(spec, None)
    return rows


def common_tags(spec: str) -> set[str]:
    # [FUNCTION DOC]
    # Returns keywords carried by at least 90% of a dataset's cached rows; empty when rows are
    # not cached or there are 3 or fewer tasks.
    """Tags nine in ten of a dataset's tasks carry (`terminal` on a terminal dataset): they tell no two
    tasks apart. Empty until the dataset's rows are cached, and for datasets of three tasks or fewer."""
    with _ROWS_LOCK:
        rows = _ROWS.get(spec) or []
    if len(rows) <= 3:
        return set()
    counts: dict[str, int] = {}
    for r in rows:
        for k in r.get("keywords") or []:
            counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= 0.9 * len(rows)}


def _config_sections(doc: dict[str, Any]) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: summarizes task.toml as labelled sections of key/value rows: Environment (image
    # or "built from Dockerfile", workdir, CPUs, memory, storage, GPUs, network, build timeout),
    # Agent and Verifier (timeout, user, network), Setup (healthcheck command length and
    # timeout), MCP servers, and names (not values) of environment/verifier env variables.
    # Empty values are omitted; times shown in s or min.
    """`task.toml` as the settings a person checks before running: where, with what, for how long."""
    env = doc.get("environment") or {}
    agent = doc.get("agent") or {}
    verifier = doc.get("verifier") or {}

    def secs(value: Any) -> str:
        # [FUNCTION DOC]
        # HELPER: formats seconds as "N s", or "N min" when >= 120; "" if not numeric.
        try:
            s = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{s / 60:.0f} min" if s >= 120 else f"{s:.0f} s"

    network = env.get("network_mode") or (
        "internet"
        if env.get("allow_internet") is True
        else "none"
        if env.get("allow_internet") is False
        else ""
    )
    image = env.get("docker_image") or "built from environment/Dockerfile"
    sections = [
        (
            "Environment",
            [
                ("Image", image),
                ("Working directory", env.get("workdir", "")),
                ("CPUs", env.get("cpus", "")),
                ("Memory", f"{env['memory_mb']:,} MB" if env.get("memory_mb") else ""),
                (
                    "Storage",
                    f"{env['storage_mb']:,} MB" if env.get("storage_mb") else "",
                ),
                ("GPUs", env.get("gpus", "")),
                ("Network", network),
                ("Build timeout", secs(env.get("build_timeout_sec"))),
            ],
        ),
        (
            "Agent",
            [
                ("Timeout", secs(agent.get("timeout_sec"))),
                ("User", agent.get("user", "")),
                ("Network", (doc.get("agent") or {}).get("network_mode", "")),
            ],
        ),
        (
            "Verifier",
            [
                ("Timeout", secs(verifier.get("timeout_sec"))),
                ("User", verifier.get("user", "")),
                ("Network", verifier.get("network_mode", "")),
            ],
        ),
    ]
    out = []
    for label, rows in sections:
        kept = [_row(k, v) for k, v in rows if v not in ("", None)]
        if kept:
            out.append({"label": label, "rows": kept})
    health = env.get("healthcheck") or {}
    if health.get("command"):
        out.append(
            {
                "label": "Setup",
                "rows": [
                    {
                        "key": "Before the agent",
                        "value": f"runs a setup command ({len(str(health['command'])):,} characters)"
                        + (
                            f", up to {secs(health['timeout_sec'])}"
                            if health.get("timeout_sec")
                            else ""
                        ),
                    }
                ],
            }
        )
    servers = env.get("mcp_servers") or []
    if servers:
        out.append(
            {
                "label": "MCP servers",
                "rows": [
                    {
                        "key": str(s.get("name", "server")),
                        "value": str(s.get("transport") or s.get("url") or ""),
                    }
                    for s in servers
                    if isinstance(s, dict)
                ],
            }
        )
    # Names only. Values are templates such as `${HF_TOKEN}` here, but a dataset could hard-code one,
    # and the UI is the wrong place to find that out.
    for label, table in (
        ("Environment variables", env.get("env")),
        ("Verifier variables", verifier.get("env")),
    ):
        if isinstance(table, dict) and table:
            out.append(
                {
                    "label": label,
                    "rows": [{"key": k, "value": ""} for k in sorted(table)],
                }
            )
    return out


def _row(key: str, value: Any) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: builds a settings row; digest-pinned images are shortened to 12 hex chars with
    # the full value kept under "full".
    """A settings row. A digest-pinned image is shortened for reading; `full` keeps it for hover."""
    text = str(value)
    if "@sha256:" in text:
        name, _, digest = text.partition("@sha256:")
        return {"key": key, "value": f"{name}@sha256:{digest[:12]}…", "full": text}
    return {"key": key, "value": text}


def _metadata(doc: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    # [FUNCTION DOC]
    # HELPER: splits [metadata] into shown {key, value} rows (lists joined, dicts skipped,
    # values truncated to 300, empty skipped) and the names of answer-like keys withheld.
    """Free-form `[metadata]` fields to show, and the answer-like ones withheld from the summary."""
    shown, hidden = [], []
    for key, value in (doc.get("metadata") or {}).items():
        if _ANSWER_KEY.search(key):
            hidden.append(key)
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            continue
        text = str(value)
        if text:
            shown.append({"key": key, "value": text[:300]})
    return shown, hidden


def file_tree(task_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    # [FUNCTION DOC]
    # Walks the resolved task directory (sorted, hidden entries skipped) and returns
    # ({path, size} for each file, truncated flag), stopping at 2000 files. Symlinks pointing
    # outside the task and unreadable files are omitted.
    """Files under a task directory as `(path, size)`, capped so a vendored repo cannot flood the page.

    Returns:
        `tuple` of the file list and whether it was cut at `MAX_TREE_FILES`.
    """
    files: list[dict[str, Any]] = []
    root = task_dir.resolve()
    # Files at the task's top level first (instruction.md, task.toml), then each folder in turn,
    # walked lazily: a vendored tree of a million files is read only as far as the cap.
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(n for n in names if not n.startswith(".")):
            if len(files) >= MAX_TREE_FILES:
                return files, True
            path = Path(folder) / name
            try:
                # a symlink out of the task can't be opened (read_task_file); don't show its size either
                if path.is_symlink() and not path.resolve().is_relative_to(root):
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            files.append({"path": path.relative_to(root).as_posix(), "size": size})
    return files, False


def task_detail(spec: str, index: int) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Builds everything the task page shows: the task row (without paragraphs) plus dataset,
    # description, full instruction text, config sections, shown metadata, withheld keys,
    # file list and truncation flag. Reads are confined to the dataset; a task outside it has
    # no files/instruction. Raises IndexError/ValueError for bad split/index.
    """Everything the task view shows for one task. Reads files, runs nothing.

    Args:
        spec (`str`):
            Dataset spec.
        index (`int`):
            Task index within the dataset.

    Returns:
        `dict` with the row fields plus `description`, `instruction`, `config`, `metadata`,
        `withheld` (answer-like metadata keys left out of the summary), `files` and
        `files_truncated`.
    """
    task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
    root = task_root(spec, task_dir)
    doc = _toml(root)
    try:
        instruction = _own(root, "instruction.md").read_text(errors="replace")
    except OSError:
        instruction = ""
    shown, hidden = _metadata(doc)
    files, truncated = file_tree(root) if root is not None else ([], False)
    row = task_row(int(index), task_dir, spec)
    row.pop("paragraphs", None)
    return {
        **row,
        "dataset": spec,
        "description": str((doc.get("task") or {}).get("description") or ""),
        "instruction": instruction,
        "config": _config_sections(doc),
        "metadata": shown,
        "withheld": hidden,
        "files": files,
        "files_truncated": truncated,
    }


def reads_environment(spec: str, index: int) -> bool:
    # [FUNCTION DOC]
    # Returns True if the task could read this server's environment variables or host files:
    # task folder outside its dataset; task.toml or environment/ resolving outside the task;
    # unreadable/invalid task.toml or any "${" in it (raw or parsed strings); any path under
    # environment/ escaping the task; any YAML file there that is too large (> 1 MB),
    # unparsable, matches a compose text pattern (variables, env_file/include/extends, host
    # bind mounts) or whose parsed content reads the host (_compose_reads). False otherwise.
    """Whether a task would read this server's environment variables or files.

    Harbor resolves `${VAR}` in `task.toml` (`[verifier.env]`, `[environment.env]`) from the process
    environment, and Docker Compose interpolates `$VAR` in a compose file the same way; an
    `environment:` or build `args:` entry that is a bare name passes the host's value straight
    through. Either one hands the task this server's keys. A compose file can also reach past its own
    folder: `env_file`, `include` and `extends` read other files, and a host path in a bind mount, a
    secret or config `file:`, a build `context`, a build cache, a watch rule or a device puts the
    host's files in the container, which a local container backend runs on this machine. So does
    asking for more of the host than a folder: `privileged`, added capabilities, the host's
    namespaces or Docker socket, another container's volumes, a named volume or network with
    settings (the local driver binds any path), or the host's SSH agent during a build.

    Both formats are checked as parsed, since that is what Harbor and Compose act on: an escape such
    as `"\\u0024{HF_TOKEN}"` has no `${` on disk. A file that doesn't parse can't be vouched for and
    counts as reading. The text patterns stay as a second net. Every YAML file under `environment/`
    is checked, since one compose file can include another.
    """
    import yaml

    root = task_root(spec, HarborTaskProvider([spec]).task_dir(spec, int(index)))
    if root is None:
        return True  # a link takes the task outside its dataset
    env_dir = root / "environment"
    toml_path = root / "task.toml"
    # A link can put any file on this machine where the task reads its own.
    if any(not p.resolve().is_relative_to(root) for p in (toml_path, env_dir)):
        return True
    if toml_path.is_file():
        try:
            text = toml_path.read_text()
            data = tomllib.loads(text)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            return True
        if _HARBOR_VAR.search(text) or any("${" in s for s in _every_string(data)):
            return True
    for path in sorted(env_dir.rglob("*")) if env_dir.is_dir() else []:
        if not path.resolve().is_relative_to(root):
            return True
        if path.suffix not in (".yml", ".yaml") or not path.is_file():
            continue
        try:
            if path.stat().st_size > _MAX_COMPOSE_BYTES:
                return True
            text = path.read_text()
            docs = list(yaml.safe_load_all(text))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return True
        if any(p.search(text) for p in _COMPOSE_READS):
            return True
        seen: set[int] = set()
        if any(_compose_reads(doc, seen) for doc in docs):
            return True
    return False


# Harbor expands only a whole value of `${VAR}` or `${VAR:-default}` (harbor.utils.env), so any `${`
# is flagged; Docker Compose also expands a bare `$VAR`.
_HARBOR_VAR = re.compile(r"\$\{")
_COMPOSE_VAR = re.compile(r"\$(\{|[A-Za-z_])")
_COMPOSE_READS = (
    _COMPOSE_VAR,
    re.compile(r"^\s*(env_file|include|extends)\s*:", re.M),
    # a bind mount from the host: `- /abs:/x`, `- ~/x:/x`, `- ../x:/x`, or `source: /abs`
    re.compile(r"^\s*-\s*[\"']?(/|~|\.\.)[^:\n]*:", re.M),
    re.compile(r"^\s*source\s*:\s*[\"']?(/|~|\.\.)", re.M),
)
_MAX_COMPOSE_BYTES = (
    1_000_000  # compose files are small; a huge one is not worth parsing
)
# long-form mount `source`, secret/config `file`, build `context` and `dockerfile`, `develop.watch` path
_PATH_KEYS = {"source", "file", "context", "dockerfile", "path"}
# Settings that hand a container more than its own folder, whatever their value: extra privileges
# (enough to mount the host's disk on a local backend), other containers' volumes, the host's SSH
# agent during a build, a file of labels read from the host. A task that needs one runs only on a
# dataset the server was started with.
_HOST_ACCESS = {
    "privileged",
    "cap_add",
    "security_opt",
    "device_cgroup_rules",
    "volumes_from",
    "ssh",
    "label_file",
    # the local volume driver binds any host path with `{type: none, o: bind, device: /etc}`
    "driver_opts",
    # the Docker API socket (the whole machine), a plugin binary run on the host, a build granted
    # the host's network or insecure mode
    "use_api_socket",
    "provider",
    "entitlements",
}
# Namespaces a container can share with the host or another container: `pid: host` alone shows it
# every process's environment on the machine, this server's included.
_NAMESPACES = {"pid", "ipc", "network_mode", "userns_mode", "uts", "cgroup", "network"}


def _outside(path: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if a compose path is absolute, under ~, a Windows drive path, or normalizes to climb out of its folder.
    """Whether a path in a compose file points outside its own folder: absolute, under `~`, a
    Windows drive, or climbing out once normalised (`foo/../../etc` is `../etc`)."""
    p = path.strip().replace("\\", "/")
    if p.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", p):
        return True
    n = posixpath.normpath(p)
    return n == ".." or n.startswith("../")


def _every_string(node: Any):
    # [FUNCTION DOC]
    # HELPER: generator yielding every string (keys and values) in a parsed document.
    """Every string in a parsed document, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _every_string(key)
            yield from _every_string(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _every_string(value)


def _passes_host_env(value: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if an environment/args entry passes host values: a bare string, a list
    # entry without "=" (or non-string), or a mapping entry with a null value.
    """An `environment:` / `args:` entry that takes its value from the host.

    A bare name in the list form (`- HF_TOKEN`) or an empty value in the map form (`HF_TOKEN:`)
    copies the host's variable in; a secret's or config's `environment: NAME` is its value.
    """
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return any(not isinstance(e, str) or "=" not in e for e in value)
    if isinstance(value, dict):
        return any(v is None for v in value.values())
    return False


def _compose_reads(node: Any, seen: set[int]) -> bool:
    # [FUNCTION DOC]
    # HELPER: recursively decides whether a parsed compose node reads host variables or files:
    # any string containing "$"; keys env_file/include/extends/devices; host-passing
    # environment/args; truthy host-access settings (privileged, cap_add, etc.); external
    # resources; host or container namespaces; named volumes with settings; networks with a
    # name or non-bridge/overlay driver; local build caches; volume entries or path keys
    # pointing outside; outside additional_contexts. Each dict/list is visited once (alias-bomb safe).
    """Whether a parsed compose document reads the host's variables or files.

    YAML aliases share nodes, so each container is walked once: an alias bomb that is tiny on disk
    would otherwise take exponential time to walk.
    """
    if isinstance(node, str):
        return "$" in node
    if not isinstance(node, (dict, list)) or id(node) in seen:
        return False
    seen.add(id(node))
    if isinstance(node, list):
        return any(_compose_reads(value, seen) for value in node)
    for key, value in node.items():
        key = str(key)
        if key in ("env_file", "include", "extends", "devices"):
            return True
        if key in ("environment", "args") and _passes_host_env(value):
            return True
        if key in _HOST_ACCESS and value not in (None, False, "", [], {}):
            return True
        if key == "external" and value:
            return True
        if (
            key in _NAMESPACES
            and isinstance(value, str)
            and (
                value.strip().lower() == "host"
                or value.strip().startswith("container:")
            )
        ):
            return True
        # A named volume is a map at the top level (a service lists its mounts), and one with any
        # setting can bind a host path or reuse a volume that already exists on the machine.
        if key == "volumes" and isinstance(value, dict):
            if any(
                isinstance(c, dict) and set(map(str, c)) - {"labels"}
                for c in value.values()
            ):
                return True
        # A network named for one that exists (`host`, another project's), or on a driver that
        # attaches to the host's interfaces (`macvlan`, `ipvlan`, `host`).
        if key == "networks" and isinstance(value, dict):
            for c in value.values():
                if isinstance(c, dict) and (
                    "name" in c
                    or str(c.get("driver", "bridge")) not in ("bridge", "overlay")
                ):
                    return True
        # a local build cache reads (`cache_from`) or writes (`cache_to`) a host folder
        if key in ("cache_from", "cache_to") and any(
            "type=local" in v.replace(" ", "")
            for v in ([value] if isinstance(value, str) else value or [])
            if isinstance(v, str)
        ):
            return True
        if (
            key == "volumes"
            and isinstance(value, list)
            and any(
                isinstance(e, str) and ":" in e and _outside(e.split(":", 1)[0])
                for e in value
            )
        ):
            return True
        if key in _PATH_KEYS and isinstance(value, str) and _outside(value):
            return True
        # the map form is `name: path`, the list form `- name=path`
        if key == "additional_contexts" and any(
            isinstance(v, str)
            and _outside(v.split("=", 1)[-1] if isinstance(value, list) else v)
            for v in (value.values() if isinstance(value, dict) else value or [])
        ):
            return True
        if _compose_reads(key, seen) or _compose_reads(value, seen):
            return True
    return False


def read_task_file(spec: str, index: int, path: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Returns a task file for the viewer: refuses (error dict) paths resolving outside the task
    # or not regular files; reads up to 256 KiB; returns binary True when a NUL byte appears
    # in the first 8 KiB, else UTF-8 text (with replacement) and whether it was truncated, plus size.
    """One file from a task directory, for the file viewer.

    The path comes from the browser, so it is resolved and checked to stay inside the task
    directory: `..` segments and symlinks pointing elsewhere are refused.

    Args:
        spec (`str`):
            Dataset spec.
        index (`int`):
            Task index.
        path (`str`):
            Path relative to the task directory.

    Returns:
        `dict` with `path`, `size`, and either `text` (possibly `truncated`) or `binary: True`, or
        `error` when the path is refused or missing.
    """
    root = task_root(spec, HarborTaskProvider([spec]).task_dir(spec, int(index)))
    target = (root / str(path)).resolve() if root is not None else None
    if target is None or not target.is_relative_to(root) or not target.is_file():
        return {"path": path, "error": "not a file in this task"}
    size = target.stat().st_size
    with target.open("rb") as fh:
        raw = fh.read(MAX_FILE_BYTES)
    if b"\x00" in raw[:8192]:
        return {"path": path, "size": size, "binary": True}
    return {
        "path": path,
        "size": size,
        "text": raw.decode("utf-8", errors="replace"),
        "truncated": size > MAX_FILE_BYTES,
    }


def search_hub(query: str = "", limit: int = 40) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Lists Hub datasets tagged "harbor" anonymously, optionally filtered by query, sorted by
    # downloads, limit clamped to 1-100. Returns id, downloads, likes and rl-environment tag
    # flag. Network call; Hub errors propagate.
    """Harbor datasets on the Hugging Face Hub, most downloaded first.

    Harbor datasets carry the `harbor` tag (and usually `rl-environment`), which is what makes them
    findable without a registry of our own.

    Args:
        query (`str`, *optional*, defaults to `""`):
            Free-text filter on the dataset id.
        limit (`int`, *optional*, defaults to `40`):
            Maximum number of results.

    Returns:
        `list[dict]` with `id`, `downloads`, `likes` and whether it is tagged `rl-environment`.
    """
    from huggingface_hub import HfApi

    # Anonymous: the server's own token would list its private datasets to any visitor.
    results = HfApi(token=False).list_datasets(
        filter="harbor",
        search=(query or "").strip() or None,
        sort="downloads",
        limit=max(1, min(int(limit), 100)),
    )
    return [
        {
            "id": d.id,
            "downloads": d.downloads or 0,
            "likes": d.likes or 0,
            "rl_environment": "rl-environment" in (d.tags or []),
        }
        for d in results
    ]


HF_ROUTER = "https://router.huggingface.co/v1"
_MODELS: tuple[float, list[dict[str, Any]]] = (0.0, [])
_MODELS_LOCK = threading.Lock()


def hf_models(ttl: float = 600.0) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Returns the Inference Providers chat model list from the HF router (cached for ttl
    # seconds, process-wide): per model id, live providers with tool support, context length
    # and prices, whether any provider supports tools, and max context; non-text-output and
    # provider-less models skipped; sorted tool-capable first then by provider count.
    # Network errors propagate.
    """Chat models on Hugging Face Inference Providers, for the model picker.

    The router's model list is public and says, per provider, whether it is live, whether it calls
    tools, its context length and its price. An agent is useless without tool calls, so each model
    carries `tools` (any live provider supports them) and the picker offers those first. Cached for
    `ttl` seconds: it is the same list for every visitor.

    Returns:
        `list[dict]` with `id`, `tools`, `context` (the largest across providers) and `providers`,
        each `{name, tools, context, price_in, price_out}` in USD per million tokens.
    """
    import json
    import time
    import urllib.request

    global _MODELS
    with _MODELS_LOCK:
        fetched, cached = _MODELS
        if cached and time.time() - fetched < ttl:
            return cached
    with urllib.request.urlopen(f"{HF_ROUTER}/models", timeout=20) as r:
        data = json.loads(r.read()).get("data") or []
    out = []
    for m in data:
        providers = []
        for p in m.get("providers") or []:
            if p.get("status", "live") != "live":
                continue
            price = p.get("pricing") or {}
            providers.append(
                {
                    "name": p.get("provider", ""),
                    "tools": bool(p.get("supports_tools")),
                    "context": p.get("context_length") or 0,
                    "price_in": price.get("input"),
                    "price_out": price.get("output"),
                }
            )
        if not providers or "text" not in (
            (m.get("architecture") or {}).get("output_modalities") or ["text"]
        ):
            continue
        out.append(
            {
                "id": m.get("id", ""),
                "tools": any(p["tools"] for p in providers),
                "context": max((p["context"] for p in providers), default=0),
                "providers": providers,
            }
        )
    # Tool-calling models first, then by how widely served, which is a fair proxy for how well used.
    out.sort(key=lambda m: (not m["tools"], -len(m["providers"])))
    with _MODELS_LOCK:
        _MODELS = (time.time(), out)
    return out


# ── adding a Hub dataset from the page ──────────────────────────────────────────────────────────
# A download of a task suite is thousands of small files and can take minutes, so an add runs in the
# background and the page polls it: first a check that the dataset is laid out the way Harbor reads
# it (before anything is downloaded), then the download with its file count, then reading the tasks.

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
MAX_ADDS_AT_ONCE = 2


def _max_add_bytes() -> int:
    # [FUNCTION DOC]
    # HELPER: returns the add size cap in bytes from OPENENV_HARBOR_UI_MAX_ADD_GB (default 5 GB; invalid -> 5 GB).
    try:
        return int(float(os.environ.get("OPENENV_HARBOR_UI_MAX_ADD_GB") or 5) * 1e9)
    except ValueError:
        return int(5e9)


def hub_summary(spec: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Anonymously inspects a Hub dataset: tasks = number of tasks/<name>/ folders whose first
    # contains task.toml, or (grouped layout) the count of task.toml files found in a bounded
    # recursive listing (None if zero or no tasks folder); bytes = the repo's used storage.
    # Non-404 Hub errors propagate.
    """What adding a Hub dataset would bring: `tasks` held as `tasks/<name>/`, the layout Harbor
    loads (`None` when it has none), and the repository's size in `bytes`.

    Read from the Hub's listing and metadata, anonymously, so nothing is downloaded to find out. The
    size is the whole repository's, an upper bound on what the `tasks/` download takes.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    api = HfApi(token=False)
    tasks = None
    try:
        folders = [
            e.path
            for e in api.list_repo_tree(
                spec, repo_type="dataset", path_in_repo="tasks", recursive=False
            )
            if isinstance(e, RepoFolder)
        ]
        if folders:
            first = {
                e.path.rsplit("/", 1)[-1]
                for e in api.list_repo_tree(
                    spec, repo_type="dataset", path_in_repo=folders[0], recursive=False
                )
            }
            if "task.toml" in first:
                tasks = len(folders)
            else:
                # Grouped (`tasks/<field>/<task>/`), which the loader searches the same way: count
                # the task files. Bounded, so a vast repository cannot stall the page.
                tasks = 0
                for i, e in enumerate(
                    api.list_repo_tree(
                        spec, repo_type="dataset", path_in_repo="tasks", recursive=True
                    )
                ):
                    tasks += e.path.endswith("/task.toml")
                    if i > 50_000:
                        break
                tasks = tasks or None
    except Exception as exc:  # noqa: BLE001 - Hub exception types vary across supported clients
        if not _hub_not_found(exc):
            raise
    return {"tasks": tasks, "bytes": api.dataset_info(spec).used_storage}


def _tasks_bytes(spec: str, cap: int, max_files: int = 500_000) -> int | None:
    # [FUNCTION DOC]
    # HELPER: sums file sizes in the dataset's tasks/ listing, returning early once above cap;
    # None if the listing fails or exceeds max_files entries.
    """The size of a Hub dataset's `tasks/` folder from its file listing, or `None` if unknown.

    Stops counting once the total passes `cap`, since that already settles the question; a listing
    longer than `max_files` is not read to the end, and counts as unknown.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    total = 0
    try:
        for n, entry in enumerate(
            HfApi(token=False).list_repo_tree(
                spec, repo_type="dataset", path_in_repo="tasks", recursive=True
            )
        ):
            if isinstance(entry, RepoFile):
                total += entry.size
            if total > cap:
                return total
            if n >= max_files:
                return None
    except Exception:  # noqa: BLE001 - no listing, no size
        return None
    return total


def _progress(job: dict[str, Any]) -> Any:
    # [FUNCTION DOC]
    # HELPER: returns a tqdm subclass that writes to a buffer and records the Hub download's
    # file-count progress (total, done) into job, ignoring byte-unit bars.
    """A silent tqdm that records the Hub download's file count in `job`.

    The Hub client uses the same class for each large file's byte counter (`unit="B"`), so only the
    bar that counts files is recorded.
    """
    import io

    from tqdm import tqdm

    class Progress(tqdm):
        # [CLASS DOC]
        # HELPER: silent tqdm recording file counts into the job dict.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # [FUNCTION DOC]
            # Redirects output to a StringIO, forces enabled, notes whether this bar counts files,
            # and stores its total in the job.
            kwargs["file"] = io.StringIO()
            kwargs.pop("disable", None)
            self._files = kwargs.get("unit", "it") != "B"
            super().__init__(*args, **kwargs)
            if self._files and self.total:
                job["total"] = int(self.total)

        def update(self, n: float | None = 1) -> bool | None:
            # [FUNCTION DOC]
            # Updates the bar and, for the file bar, stores the current count in the job.
            out = super().update(n)
            if self._files:
                job["done"] = int(self.n)
            return out

    return Progress


def added_spec(spec: str, settings: Any) -> str:
    # [FUNCTION DOC]
    # Returns the loader spec for a dataset added from the page: the bucket mount folder
    # (<mount>/<org>__<name>) when a bucket is mounted, else the Hub id.
    """What the loader is given for a Hub dataset added from the page.

    With the Space's bucket mounted, the dataset's folder on the mount, the same place `push` puts
    the datasets it serves; otherwise the Hub id, which downloads to this machine's cache.
    """
    if settings.bucket and settings.bucket_mount:
        return str(settings.bucket_mount / spec.replace("/", "__"))
    return spec


def hub_id(spec: str) -> str:
    # [FUNCTION DOC]
    # Converts a "/.../org__name" mount path back to "org/name"; other specs are returned unchanged.
    """The Hub id a dataset spec stands for: `org/name` for a `/data/org__name` mount folder."""
    name = Path(spec).name if spec.startswith("/") else ""
    return name.replace("__", "/", 1) if "__" in name else spec


def added_in_bucket(settings: Any, served: list[str]) -> list[str]:
    # [FUNCTION DOC]
    # Lists bucket mount folders named org__name with a tasks/ directory that are not served
    # datasets (by path or Hub id) and contain no linked task folders. Empty without a mounted bucket.
    """Datasets on the mounted bucket that the server was not started with: the ones added from the
    page. The bucket itself is the list, so it survives restarts and needs no file of its own."""
    mount = settings.bucket_mount
    if not (settings.bucket and mount):
        return []
    out = []
    for p in sorted(mount.iterdir()):
        if (
            p.is_dir()
            and "__" in p.name
            and not p.name.startswith(".")
            and (p / "tasks").is_dir()
        ):
            # Task folders that are links stay out, e.g. a refused add a failed cleanup left behind.
            if (
                str(p) not in served
                and hub_id(str(p)) not in served
                and not _links_a_task(p)
            ):
                out.append(str(p))
    return out


def _forget(spec: str) -> None:
    # [FUNCTION DOC]
    # HELPER: drops a dataset from the task-dir cache and the UI row caches.
    from . import tasks

    with tasks._LOCK:
        tasks._CACHE.pop(spec, None)
    with _ROWS_LOCK:
        _ROWS.pop(spec, None)
        _ROWS_INFLIGHT.pop(spec, None)


def remove_added(spec: str, settings: Any) -> None:
    # [FUNCTION DOC]
    # Deletes a dataset added from the page: for a bucket mount folder (must sit directly in
    # the mount and be named org__name, else ValueError) deletes all its bucket files in
    # batches of 1000 via the Hub API; otherwise removes the local download folder in the
    # dataset cache (must sit directly in it, else ValueError). Then clears caches.
    """Delete a dataset added from the page: its folder in the bucket, or its local download.

    Only ever a folder this module put there: one under the bucket mount named `org__name`, or one in
    the dataset cache. The caller checks that `spec` was added from the page, not served.
    """
    from . import tasks

    if spec.startswith("/") and settings.bucket and settings.bucket_mount:
        folder = Path(spec).resolve()
        if folder.parent != settings.bucket_mount.resolve() or "__" not in folder.name:
            raise ValueError("not a dataset folder on the bucket")
        from huggingface_hub import BucketFile, HfApi

        api = HfApi()
        paths = [
            f.path
            for f in api.list_bucket_tree(
                settings.bucket, prefix=f"{folder.name}/", recursive=True
            )
            if isinstance(f, BucketFile)
        ]
        for i in range(0, len(paths), 1000):
            api.batch_bucket_files(settings.bucket, delete=paths[i : i + 1000])
    else:
        root = tasks._DATASET_ROOT.resolve()
        folder = (tasks._DATASET_ROOT / spec.replace("/", "__")).resolve()
        if folder.parent != root:
            raise ValueError("not a downloaded dataset")
        if folder.is_dir():
            import shutil

            shutil.rmtree(folder)
    _forget(spec)


def _copy_to_bucket(
    spec: str, settings: Any, job: dict[str, Any], expected: int
) -> str:
    # [FUNCTION DOC]
    # HELPER: server-side copies the dataset's tasks/ into the bucket under org__name/tasks/,
    # then polls the mount every 3s for up to 180s until it shows at least the expected number
    # of tasks; returns the folder path or raises ValueError on timeout. Updates job state.
    """Copy a Hub dataset into the Space's bucket, server side, and wait for the mount to show it.

    The same copy `push` makes for the datasets it serves: by content hash, so nothing is downloaded
    or uploaded and a suite of thousands of files takes seconds. Only `tasks/` is copied: it is all
    the loader reads, the same folder a download fetches, and what the size check measured. The
    mount then shows the new folder, which is waited for rather than assumed.
    """
    import time

    from huggingface_hub import HfApi

    prefix = spec.replace("/", "__")
    job["state"] = "copying"
    HfApi().copy_files(
        f"hf://datasets/{spec}/tasks/",
        f"hf://buckets/{settings.bucket}/{prefix}/tasks/",
    )
    job["state"] = "mounting"
    folder = settings.bucket_mount / prefix
    deadline = time.time() + 180
    from .tasks import _task_dirs_from_directory

    while time.time() < deadline:
        try:
            if (folder / "tasks").is_dir() and len(
                _task_dirs_from_directory(folder)
            ) >= expected:
                return str(folder)
        except OSError:
            pass
        time.sleep(3)
    raise ValueError(
        "Copied to the bucket, but the mount has not shown it yet. It appears when the Space restarts."
    )


def start_add(spec: str, on_added: Any, settings: Any = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Starts (or returns the in-progress) background add of a Hub dataset. Refuses with an
    # error dict when MAX_ADDS_AT_ONCE adds are running. The thread checks the layout (error
    # without tasks), size (Hub size or measured; error if unknown or above the cap), copies
    # into the bucket or downloads with progress, refuses and removes the copy if it contains
    # any symlink, indexes the tasks, calls on_added(target) and marks done; any exception is
    # recorded in the job as an error (message max 240 chars). Returns a copy of the job dict.
    """Start adding `spec` in the background, or return the add already under way.

    `on_added(loader_spec)` is called once its tasks are read. The caller has checked that `spec` is a
    public Hub dataset id; this checks its layout and size, puts it where `added_spec` says (the
    bucket, or this machine), and indexes it.
    """
    import time

    with _JOBS_LOCK:
        job = _JOBS.get(spec)
        if job and job["state"] not in ("error", "done"):
            return dict(job)
        busy = sum(1 for j in _JOBS.values() if j["state"] not in ("error", "done"))
        if busy >= MAX_ADDS_AT_ONCE:
            return {
                "spec": spec,
                "state": "error",
                "error": "Another dataset is being added. Try again in a moment.",
            }
        job = {
            "spec": spec,
            "state": "checking",
            "done": 0,
            "total": 0,
            "tasks": None,
            "error": None,
            "started": time.time(),
        }
        _JOBS[spec] = job

    def run() -> None:
        # [FUNCTION DOC]
        # HELPER: thread body performing the add steps described above.
        try:
            summary = hub_summary(spec)
            expected = summary["tasks"]
            if not expected:
                raise ValueError(
                    "This dataset has no tasks/<name>/ folder, which is how Harbor datasets are laid out."
                )
            if not summary["bytes"]:
                # The Hub has no size for some repositories, and reports 0 for one it has not
                # measured yet (a fresh upload): measure what the download would take.
                summary["bytes"] = _tasks_bytes(spec, _max_add_bytes())
            if summary["bytes"] is None:
                raise ValueError(
                    "Couldn't tell how big this dataset is, so it isn't added from the page."
                )
            if summary["bytes"] > _max_add_bytes():
                raise ValueError(
                    f"This dataset is {summary['bytes'] / 1e9:.1f} GB, over the "
                    f"{_max_add_bytes() / 1e9:.0f} GB this server adds from the page "
                    "(OPENENV_HARBOR_UI_MAX_ADD_GB)."
                )
            job["bytes"] = summary["bytes"]
            job["expected"] = expected
            if settings is not None and settings.bucket and settings.bucket_mount:
                target = _copy_to_bucket(spec, settings, job, expected)
            else:
                job["state"] = "downloading"
                resolve_task_dirs(spec, tqdm_class=_progress(job))
                target = spec
            # A link in a dataset from the Hub can point a task, or the files the page shows by
            # itself, at anything on this machine. Refused, and removed so a restart, which lists
            # the bucket's folders, does not bring it back.
            from . import tasks as _tasks

            root = (
                Path(target)
                if target.startswith("/")
                else _tasks._DATASET_ROOT / target.replace("/", "__")
            )
            link = _first_link(root)
            if link is not None:
                refusal = (
                    f"This dataset contains a symbolic link ({link.relative_to(root)}), which "
                    "a dataset added from the page may not."
                )
                try:
                    remove_added(target, settings or _NO_BUCKET)
                except Exception as exc:  # noqa: BLE001 - the refusal stands either way
                    # What was copied stays. A restart skips it if its task folders are links
                    # (`added_in_bucket`), and no link further down is ever followed.
                    refusal += f" Removing the copy failed ({type(exc).__name__})."
                raise ValueError(refusal)
            job["state"] = "indexing"
            job["tasks"] = len(task_rows(target))
            job["target"] = target
            on_added(target)
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - reported to the page, never raised
            job.update(state="error", error=f"{str(exc)[:240]}")

    threading.Thread(target=run, daemon=True, name=f"harbor-add-{spec}").start()
    return dict(job)


# `remove_added`'s settings for a server with no bucket: the dataset is a local download
_NO_BUCKET = type("NoBucket", (), {"bucket": None, "bucket_mount": None})()


def add_status(spec: str) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Returns a copy of the add job for spec, or None.
    with _JOBS_LOCK:
        job = _JOBS.get(spec)
        return dict(job) if job else None

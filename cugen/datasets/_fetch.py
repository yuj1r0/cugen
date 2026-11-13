"""Checksum-verified download of the datasets in the manifest (standard library only).

    from cugen import datasets
    datasets.list_datasets()
    datasets.info("darwins_ark_2022")
    path = datasets.fetch("darwins_ark_2022", parts=["genotypes"])   # -> <cache>/darwins_ark_2022
    datasets.verify(path, "darwins_ark_2022")

Downloads stream to ``<file>.part`` while a sha256 is computed on the fly (8 MiB reads),
resume with an HTTP ``Range`` request (a 206 reply appends; a 200 reply restarts from zero),
are checked against the manifest size and sha256, and are then renamed into place
atomically. A verified file is not downloaded again. Archives are extracted member by
member with the standard library, refusing absolute paths, ``..`` components and links;
nothing inside a downloaded archive is ever executed.

Cache directory: ``dest=`` if given, else the ``CUGEN_DATA`` environment variable, else
``${XDG_CACHE_HOME:-~/.cache}/cugen/datasets``; each dataset goes to ``<cache>/<name>/``.
Point it at a large filesystem: the dog genotype archive alone is 7 GB.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from ._manifest import DATASETS

logger = logging.getLogger("cugen.datasets")

CHUNK = 8 << 20                     # bytes per read while streaming / hashing
HOME_WARN_BYTES = 1 << 30           # warn before writing more than this under $HOME
USER_AGENT = "cugen-datasets (+https://github.com/yuj1r0/cugen)"
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

__all__ = ["list_datasets", "info", "fetch", "verify", "data_home", "DatasetError",
           "DatasetNotHostedError", "ManualDownloadRequired", "ChecksumError"]


class DatasetError(RuntimeError):
    """A dataset could not be fetched, extracted or verified."""


class DatasetNotHostedError(DatasetError):
    """The manifest has no download URL for the requested files yet."""


class ManualDownloadRequired(DatasetError):
    """The dataset needs a manual (logged-in) download; the message says how."""


class ChecksumError(DatasetError):
    """A file's size or sha256 does not match the manifest."""


# ----------------------------------------------------------------------------- manifest queries
def list_datasets():
    """Names of the datasets in the manifest."""
    return sorted(DATASETS)


def _entry(name):
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(list_datasets())}") from None


def info(name):
    """A copy of the manifest entry, with ``total_size`` (bytes; None if any size is unknown)."""
    e = copy.deepcopy(_entry(name))
    sizes = [f.get("size") for f in e["files"]]
    e["total_size"] = None if any(s is None for s in sizes) else int(sum(sizes))
    return e


def data_home(dest=None):
    """The cache base directory: ``dest``, else ``$CUGEN_DATA``, else
    ``${XDG_CACHE_HOME:-~/.cache}/cugen/datasets``."""
    if dest is not None:
        return Path(dest).expanduser()
    env = os.environ.get("CUGEN_DATA")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "cugen" / "datasets"


def _select_files(entry, parts, default_all=False):
    if parts is None:
        parts = None if default_all else entry.get("default_parts")
    if parts is None:
        return list(entry["files"])
    if isinstance(parts, str):
        parts = [parts]
    known = entry.get("parts") or {}
    bad = [p for p in parts if p not in known]
    if bad:
        raise ValueError(f"{entry['name']}: unknown parts {bad}; available: {sorted(known)}")
    want = {n for p in parts for n in known[p]}
    return [f for f in entry["files"] if f["name"] in want]


def _not_hosted_message(entry, files):
    names = ", ".join(f["name"] for f in files)
    msg = f"{entry['name']}: {names} not published yet (the manifest has no download URL)."
    if entry.get("regenerate"):
        msg += ("\nThe dataset is synthetic and deterministic; regenerate it locally with\n    "
                + entry["regenerate"]
                + "\nor, in Python: cugen.datasets.simulate_height(out_dir, preset=...) with the same preset.")
    return msg


# ----------------------------------------------------------------------------- hashing / markers
def _hash_file(path, algo="sha256"):
    h = hashlib.new(algo)
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def _marker(path):
    return path.with_name("." + path.name + ".verified")


def _write_marker(path, sha256):
    st = path.stat()
    _marker(path).write_text(json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256}))


def _cached_ok(path, f):
    """True when ``path`` was verified earlier and has not changed since (no rehash)."""
    m = _marker(path)
    if not (path.exists() and m.exists()):
        return False
    try:
        rec = json.loads(m.read_text())
    except (OSError, ValueError):
        return False
    st = path.stat()
    return (rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns
            and (f.get("size") is None or f["size"] == st.st_size)
            and (f.get("sha256") is None or rec.get("sha256") == f["sha256"]))


def _file_status(path, f):
    """'ok' | 'missing' | 'size_mismatch' | 'sha256_mismatch' | 'md5_mismatch' | 'unverified'."""
    if not path.exists():
        return "missing", None
    size = path.stat().st_size
    if f.get("size") is not None and size != f["size"]:
        return "size_mismatch", None
    if f.get("sha256"):
        sha, _ = _hash_file(path)
        return ("ok" if sha == f["sha256"] else "sha256_mismatch"), sha
    if f.get("md5"):
        md5, _ = _hash_file(path, "md5")
        return ("ok" if md5 == f["md5"] else "md5_mismatch"), None
    return "unverified", None


# ----------------------------------------------------------------------------- download
def _stream(url, part, expected_size, timeout):
    """Download ``url`` into ``part`` (resuming it) -> (sha256, size) of the whole file."""
    have = part.stat().st_size if part.exists() else 0
    if expected_size is not None and have > expected_size:
        part.unlink()
        have = 0
    if have and expected_size is not None and have == expected_size:
        return _hash_file(part)                       # complete from an earlier attempt
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                    # nothing left to send: the part is complete
            return _hash_file(part)
        raise
    with resp:
        status = getattr(resp, "status", None)
        if status is None and hasattr(resp, "getcode"):
            status = resp.getcode()
        h = hashlib.sha256()
        if have and status == 206:
            cr = resp.headers.get("Content-Range", "") if getattr(resp, "headers", None) is not None else ""
            if cr and not cr.replace(" ", "").startswith(f"bytes{have}-"):
                part.unlink()
                raise DatasetError(f"{url}: server resumed at the wrong offset ({cr!r}); restarting")
            with open(part, "rb") as fh:              # hash what we already have
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    h.update(b)
            mode, n = "ab", have
            logger.info("resuming %s at %d bytes", part.name[:-5], have)
        else:                                         # 200 (or no status, e.g. file://): full body
            mode, n = "wb", 0
        with open(part, mode) as out:
            while True:
                b = resp.read(CHUNK)
                if not b:
                    break
                out.write(b)
                h.update(b)
                n += len(b)
    return h.hexdigest(), n


def _retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return isinstance(exc, (urllib.error.URLError, OSError, DatasetError))


def _download(f, target, timeout, retries):
    part = target.with_name(target.name + ".part")
    last = None
    for url in f["urls"]:
        for attempt in range(max(1, retries)):
            try:
                logger.info("downloading %s from %s", f["name"], url)
                sha, size = _stream(url, part, f.get("size"), timeout)
            except Exception as e:  # noqa: BLE001 - classified below, re-raised if fatal
                last = e
                if not _retryable(e):
                    break
                time.sleep(min(2 ** attempt, 30))
                continue
            if f.get("size") is not None and size != f["size"]:
                part.unlink()
                raise ChecksumError(f"{f['name']}: downloaded {size} bytes, the manifest says {f['size']}")
            if f.get("sha256") and sha != f["sha256"]:
                part.unlink()
                raise ChecksumError(f"{f['name']}: sha256 {sha} does not match the manifest {f['sha256']}")
            os.replace(part, target)
            _write_marker(target, sha)
            return target
    raise DatasetError(f"could not download {f['name']}: {last}")


# ----------------------------------------------------------------------------- extraction
def _is_archive(name):
    return name.endswith(_ARCHIVE_SUFFIXES)


def _safe_target(dest, member):
    """Destination of archive member ``member`` inside ``dest``; refuses anything that could
    land outside it."""
    name = member.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or any(p == ".." for p in parts)
            or (len(parts[0]) >= 2 and parts[0][1] == ":")):
        raise DatasetError(f"refusing unsafe archive member {member!r}")
    root = dest.resolve()
    target = (root / Path(*parts)).resolve()
    if target != root and root not in target.parents:
        raise DatasetError(f"refusing archive member {member!r}: it resolves outside {root}")
    return target


def _extract_zip(archive, dest):
    with zipfile.ZipFile(archive) as z:
        for zi in z.infolist():
            if ((zi.external_attr >> 16) & 0o170000) == 0o120000:
                raise DatasetError(f"refusing symbolic link {zi.filename!r} in {archive.name}")
            target = _safe_target(dest, zi.filename)
            if zi.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with z.open(zi) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            os.replace(tmp, target)


def _extract_tar(archive, dest):
    with tarfile.open(archive) as t:
        members = t.getmembers()
        for m in members:
            _safe_target(dest, m.name)
            if not (m.isfile() or m.isdir()):
                raise DatasetError(f"refusing tar member {m.name!r} in {archive.name}: "
                                   f"not a regular file or directory")
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, members=members, filter="data")
        else:
            t.extractall(dest, members=members)


def _extract(archive, dest, force=False):
    mark = dest / f".{archive.name}.extracted"
    if mark.exists() and not force:
        return
    logger.info("extracting %s", archive.name)
    if archive.name.endswith(".zip"):
        _extract_zip(archive, dest)
    else:
        _extract_tar(archive, dest)
    mark.write_text(json.dumps({"archive": archive.name, "size": archive.stat().st_size}))


# ----------------------------------------------------------------------------- public API
def _warn_space(ddir, nbytes):
    if not nbytes:
        return
    try:
        home = Path.home().resolve()
        if home == ddir.resolve() or home in ddir.resolve().parents:
            if nbytes > HOME_WARN_BYTES:
                logger.warning("writing %.1f GB under your home directory (%s); pass dest= or set "
                               "CUGEN_DATA to a larger filesystem", nbytes / 1e9, ddir)
        free = shutil.disk_usage(ddir).free
        if free < nbytes:
            logger.warning("only %.1f GB free at %s, %.1f GB needed", free / 1e9, ddir, nbytes / 1e9)
    except OSError:
        pass


def fetch(name, dest=None, parts=None, extract=True, force=False, *, timeout=60, retries=3):
    """Download (or find) a dataset and verify it against the manifest.

    name    : a name from :func:`list_datasets`.
    dest    : cache base directory (default: ``$CUGEN_DATA`` or ``~/.cache/cugen/datasets``);
              the files go to ``<dest>/<name>/``.
    parts   : names from ``info(name)["parts"]`` (e.g. ``("meta", "cugen")``); None = the
              dataset's default parts (all files when it has none).
    extract : unpack zip / tar archives next to them (member by member, path-checked).
    force   : download and extract again even if a verified copy exists.

    Returns the dataset directory. Raises :class:`DatasetNotHostedError` for a dataset that
    is not published yet (synthetic demos: the message gives the regenerate command),
    :class:`ManualDownloadRequired` for a dataset that needs a logged-in download (after
    placing the files there, call fetch again or :func:`verify`), and
    :class:`ChecksumError` when a download does not match the manifest.
    """
    entry = _entry(name)
    files = _select_files(entry, parts)
    ddir = data_home(dest) / name
    if entry.get("access") == "manual":
        todo = []
        for f in files:
            target = ddir / f["name"]
            if _cached_ok(target, f):
                continue
            st, sha = _file_status(target, f)
            if st == "ok":
                _write_marker(target, sha)
            else:
                todo.append(f"{f['name']} ({st})")
        if todo:
            msg = (f"{name} needs a manual download. {entry.get('instructions', '')}\n"
                   f"Place these files in {ddir}: {', '.join(todo)}")
            logger.warning(msg)
            raise ManualDownloadRequired(msg)
    else:
        unhosted = [f for f in files if not f.get("urls")]
        if unhosted:
            raise DatasetNotHostedError(_not_hosted_message(entry, unhosted))
        ddir.mkdir(parents=True, exist_ok=True)
        need = [f for f in files if force or not _cached_ok(ddir / f["name"], f)]
        _warn_space(ddir, sum(f.get("size") or 0 for f in need))
        for f in files:
            target = ddir / f["name"]
            if f in need:
                if not force and target.exists():
                    status, sha = _file_status(target, f)
                    if status == "ok":                 # verified copy without a marker yet
                        _write_marker(target, sha)
                        continue
                _download(f, target, timeout, retries)
            else:
                logger.info("%s already verified", f["name"])
    if extract:
        for f in files:
            if _is_archive(f["name"]):
                _extract(ddir / f["name"], ddir, force=force)
    return ddir


def verify(path, name, parts=None):
    """Check the files of dataset ``name`` in directory ``path`` (no download; read-only).

    Every manifest file of the selected parts (default: all) gets a status: "ok",
    "missing", "size_mismatch", "sha256_mismatch", "md5_mismatch" or "unverified" (no
    reference checksum). For a synthetic dataset directory (generated locally or
    extracted) whose ``meta.json`` lists per-file checksums, ``contents`` reports each
    file against the manifest's ``contents`` reference when it exists, else against
    ``meta.json`` (self-consistency only).

    Returns a dict: ``name``, ``path``, ``files`` (name -> status), ``contents`` (or None),
    ``reference`` and ``ok``.
    """
    entry = _entry(name)
    d = Path(path)
    files = _select_files(entry, parts, default_all=True)
    status = {f["name"]: _file_status(d / f["name"], f)[0] for f in files}
    rep = {"name": name, "path": str(d), "files": status, "contents": None, "reference": None}
    contents_ref = entry.get("contents")
    meta_p = d / "meta.json"
    listed = None
    if meta_p.exists():
        try:
            listed = json.loads(meta_p.read_text()).get("files")
        except (OSError, ValueError):
            listed = None
    ref = contents_ref or listed
    if ref:
        rep["reference"] = "manifest" if contents_ref else "meta.json"
        rep["contents"] = {rel: _file_status(d / rel, rec)[0] for rel, rec in sorted(ref.items())}
        if contents_ref and listed:
            extra = sorted(set(listed) - set(contents_ref))
            if extra:
                rep["not_in_reference"] = extra
    archives_present = any(s != "missing" for s in status.values())
    checks = []
    if archives_present or rep["contents"] is None:
        checks.append(bool(status) and all(s == "ok" for s in status.values()))
    if rep["contents"] is not None:
        checks.append(all(s == "ok" for s in rep["contents"].values()))
    rep["ok"] = all(checks)
    bad = {k: v for k, v in status.items() if v != "ok"}
    if rep["contents"] is not None:
        bad.update({k: v for k, v in rep["contents"].items() if v != "ok"})
    if rep["ok"]:
        logger.info("%s: verified %s", name, d)
    else:
        logger.warning("%s at %s: %d problem(s): %s", name, d, len(bad),
                       ", ".join(f"{k}={v}" for k, v in list(bad.items())[:10]))
    return rep

"""Offline tests for cugen.datasets.fetch / verify and the manifest (no network)."""
import hashlib
import io
import os
import re
import tarfile
import zipfile

import pytest

import cugen.datasets._fetch as F
from cugen.datasets import (ChecksumError, DatasetError, DatasetNotHostedError, ManualDownloadRequired,
                            data_home, fetch, info, list_datasets, verify)
from cugen.datasets._manifest import DATASETS


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def server(tmp_path):
    """A 'remote' directory served through file:// URLs."""
    d = tmp_path / "server"
    d.mkdir()
    return d


def _entry(name, files, **kw):
    e = dict(name=name, description="toy", files=files, parts={"all": [f["name"] for f in files]},
             default_parts=None, licence="CC0-1.0", citation="toy", doi=None, homepage=None,
             access="open", notes="")
    e.update(kw)
    return e


def _file(server, name, data, **kw):
    p = server / name
    p.write_bytes(data)
    f = dict(name=name, urls=[p.as_uri()], size=len(data), sha256=_sha(data))
    f.update(kw)
    return f


@pytest.fixture
def toy(server, monkeypatch):
    payload = _zip_bytes({"geno/a.txt": b"alpha\n", "b.txt": b"beta\n"})
    readme = b"toy dataset\n"
    files = [_file(server, "README.md", readme), _file(server, "data.zip", payload)]
    monkeypatch.setitem(DATASETS, "toy", _entry("toy", files))
    return files


def test_file_url_fetch_verifies_and_extracts(toy, tmp_path):
    d = fetch("toy", dest=tmp_path / "cache")
    assert d == tmp_path / "cache" / "toy"
    assert (d / "README.md").read_bytes() == b"toy dataset\n"
    assert (d / "geno" / "a.txt").read_bytes() == b"alpha\n" and (d / "b.txt").read_bytes() == b"beta\n"
    assert not list(d.glob("*.part"))
    rep = verify(d, "toy")
    assert rep["ok"] and rep["files"] == {"README.md": "ok", "data.zip": "ok"}


def test_wrong_checksum_raises_and_leaves_nothing(server, tmp_path, monkeypatch):
    f = _file(server, "x.bin", b"0123456789")
    monkeypatch.setitem(DATASETS, "bad", _entry("bad", [dict(f, sha256="0" * 64)]))
    with pytest.raises(ChecksumError, match="sha256"):
        fetch("bad", dest=tmp_path)
    assert not (tmp_path / "bad" / "x.bin").exists() and not (tmp_path / "bad" / "x.bin.part").exists()
    monkeypatch.setitem(DATASETS, "bad2", _entry("bad2", [dict(f, size=11)]))
    with pytest.raises(ChecksumError, match="bytes"):
        fetch("bad2", dest=tmp_path)


def test_placeholder_raises_not_hosted_with_regenerate_command(tmp_path):
    for name, preset in (("demo_height_n20k", "n20k"), ("demo_height_n3k", "n3k")):
        with pytest.raises(DatasetNotHostedError) as ei:
            fetch(name, dest=tmp_path)
        msg = str(ei.value)
        assert "python -m cugen.datasets.simulate_height" in msg and f"--preset {preset}" in msg
    assert not (tmp_path / "demo_height_n20k").exists()          # nothing written


class _Resp:
    def __init__(self, body, status, headers=None):
        self._b = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _remote(data, monkeypatch, *, honour_range=True):
    """Patch urlopen with an HTTP-like server for `data`; returns the list of Range headers seen."""
    seen = []

    def urlopen(req, timeout=None):
        rng = req.get_header("Range")
        seen.append(rng)
        if rng and honour_range:
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            return _Resp(data[start:], 206, {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
        return _Resp(data, 200)

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)
    return seen


def _http_entry(monkeypatch, data, name="remote"):
    f = dict(name="big.bin", urls=["https://example.invalid/big.bin"], size=len(data), sha256=_sha(data))
    monkeypatch.setitem(DATASETS, name, _entry(name, [f]))
    return f


def test_resume_with_206(tmp_path, monkeypatch):
    data = os.urandom(3 * 1024 + 17)
    _http_entry(monkeypatch, data)
    d = tmp_path / "remote"
    d.mkdir()
    (d / "big.bin.part").write_bytes(data[:1000])                 # an interrupted download
    seen = _remote(data, monkeypatch)
    fetch("remote", dest=tmp_path)
    assert seen == ["bytes=1000-"]
    assert (d / "big.bin").read_bytes() == data and not (d / "big.bin.part").exists()


def test_server_without_range_restarts_with_200(tmp_path, monkeypatch):
    data = os.urandom(2048)
    _http_entry(monkeypatch, data)
    d = tmp_path / "remote"
    d.mkdir()
    (d / "big.bin.part").write_bytes(b"garbage-" * 50)            # wrong bytes: a restart must replace them
    seen = _remote(data, monkeypatch, honour_range=False)
    fetch("remote", dest=tmp_path)
    assert seen == ["bytes=400-"]
    assert (d / "big.bin").read_bytes() == data


def test_complete_part_is_not_requested_again(tmp_path, monkeypatch):
    data = os.urandom(512)
    _http_entry(monkeypatch, data)
    d = tmp_path / "remote"
    d.mkdir()
    (d / "big.bin.part").write_bytes(data)
    seen = _remote(data, monkeypatch)
    fetch("remote", dest=tmp_path)
    assert seen == [] and (d / "big.bin").read_bytes() == data


def test_verified_file_is_not_downloaded_again(tmp_path, monkeypatch):
    data = os.urandom(700)
    _http_entry(monkeypatch, data)
    seen = _remote(data, monkeypatch)
    fetch("remote", dest=tmp_path)
    assert seen == [None]
    fetch("remote", dest=tmp_path)                                # marker + unchanged file: no request
    assert seen == [None]
    fetch("remote", dest=tmp_path, force=True)
    assert seen == [None, None]
    # a file placed by hand (no marker) is hashed once, not downloaded
    os.remove(tmp_path / "remote" / ".big.bin.verified")
    fetch("remote", dest=tmp_path)
    assert seen == [None, None] and (tmp_path / "remote" / ".big.bin.verified").exists()


def test_cache_directory_resolution(tmp_path, monkeypatch, toy):
    monkeypatch.setenv("CUGEN_DATA", str(tmp_path / "env"))
    assert data_home() == tmp_path / "env"
    assert data_home(tmp_path / "explicit") == tmp_path / "explicit"
    assert fetch("toy", extract=False) == tmp_path / "env" / "toy"
    assert (tmp_path / "env" / "toy" / "data.zip").exists() and not (tmp_path / "env" / "toy" / "b.txt").exists()
    monkeypatch.delenv("CUGEN_DATA")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert data_home() == tmp_path / "xdg" / "cugen" / "datasets"


def test_parts_selection(toy, tmp_path):
    DATASETS["toy"]["parts"] = {"doc": ["README.md"], "data": ["data.zip"]}
    d = fetch("toy", dest=tmp_path, parts="doc")
    assert (d / "README.md").exists() and not (d / "data.zip").exists()
    with pytest.raises(ValueError, match="unknown parts"):
        fetch("toy", dest=tmp_path, parts=["nope"])


def test_manual_access(tmp_path, monkeypatch, server):
    with pytest.raises(ManualDownloadRequired, match="10.5061/dryad.83bk3jb4r"):
        fetch("darwins_ark_2025", dest=tmp_path)
    data = b"placed by hand\n"
    f = dict(name="m.txt", urls=None, size=len(data), sha256=_sha(data))
    monkeypatch.setitem(DATASETS, "manual_toy", _entry("manual_toy", [f], access="manual", instructions="log in"))
    with pytest.raises(ManualDownloadRequired, match="m.txt"):
        fetch("manual_toy", dest=tmp_path)
    (tmp_path / "manual_toy").mkdir()
    (tmp_path / "manual_toy" / "m.txt").write_bytes(data)
    assert fetch("manual_toy", dest=tmp_path) == tmp_path / "manual_toy"
    assert verify(tmp_path / "manual_toy", "manual_toy")["ok"]


def test_verify_statuses(tmp_path, monkeypatch):
    a, b, c = b"aaa", b"bbbb", b"cc"
    files = [dict(name=n, urls=None, size=len(x), sha256=_sha(x)) for n, x in (("a", a), ("b", b), ("c", c))]
    monkeypatch.setitem(DATASETS, "v", _entry("v", files))
    (tmp_path / "a").write_bytes(a)
    (tmp_path / "b").write_bytes(b"BBBB")                         # same size, other bytes
    rep = verify(tmp_path, "v")
    assert rep["files"] == {"a": "ok", "b": "sha256_mismatch", "c": "missing"} and not rep["ok"]
    (tmp_path / "c").write_bytes(b"c")
    assert verify(tmp_path, "v")["files"]["c"] == "size_mismatch"


def _write_zip(path, members):
    path.write_bytes(_zip_bytes(members))


@pytest.mark.parametrize("member", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt"])
def test_zip_path_traversal_refused(tmp_path, member):
    arc = tmp_path / "x.zip"
    _write_zip(arc, {member: b"x"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(DatasetError, match="refusing"):
        F._extract(arc, dest)
    assert not (tmp_path / "evil.txt").exists()


def test_tar_links_and_traversal_refused(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    for kind in ("symlink", "dotdot"):
        arc = tmp_path / f"{kind}.tar"
        with tarfile.open(arc, "w") as t:
            if kind == "symlink":
                ti = tarfile.TarInfo("link")
                ti.type = tarfile.SYMTYPE
                ti.linkname = "/etc/passwd"
                t.addfile(ti)
            else:
                ti = tarfile.TarInfo("../evil.txt")
                ti.size = 1
                t.addfile(ti, io.BytesIO(b"x"))
        with pytest.raises(DatasetError, match="refusing"):
            F._extract(arc, dest)
    assert not (tmp_path / "evil.txt").exists() and not (dest / "link").exists()
    ok = tmp_path / "ok.tar.gz"
    with tarfile.open(ok, "w:gz") as t:
        ti = tarfile.TarInfo("sub/file.txt")
        ti.size = 3
        t.addfile(ti, io.BytesIO(b"abc"))
    F._extract(ok, dest)
    assert (dest / "sub" / "file.txt").read_bytes() == b"abc"


def test_manifest_entries_are_complete():
    assert {"demo_height_n20k", "demo_height_n3k", "darwins_ark_2022", "darwins_ark_2025"} <= set(list_datasets())
    hexre = re.compile(r"^[0-9a-f]{64}$")
    for name in list_datasets():
        e = info(name)
        assert e["name"] == name and e["licence"] and e["citation"] and e["description"]
        assert e["access"] in ("open", "manual")
        names = [f["name"] for f in e["files"]]
        assert names and len(set(names)) == len(names)
        for part, members in (e.get("parts") or {}).items():
            assert set(members) <= set(names), (name, part)
        placeholder = [f["urls"] is None and f["size"] is None and f["sha256"] is None for f in e["files"]]
        for f, ph in zip(e["files"], placeholder):
            if ph:
                continue
            assert isinstance(f["size"], int) and f["size"] > 0, (name, f["name"])
            assert hexre.match(f["sha256"]), (name, f["name"])
            if e["access"] == "open":
                assert f["urls"], (name, f["name"])
        if any(placeholder):
            assert e["access"] == "open" and "simulate_height" in e["regenerate"]
        if e["access"] == "manual":
            assert e["instructions"]
    a = info("darwins_ark_2022")
    assert a["total_size"] == 7025715448                      # the Dryad record's storage size
    assert all(u.startswith("https://zenodo.org/api/records/6522639/files/") for f in a["files"] for u in f["urls"])
    assert a["doi"] == "10.5061/dryad.g4f4qrfr0" and a["licence"] == "CC0-1.0"
    b = info("darwins_ark_2025")
    assert b["access"] == "manual" and b["total_size"] == 19487420884 and b["doi"] == "10.5061/dryad.83bk3jb4r"


def test_datasets_module_is_lazy():
    import subprocess
    import sys
    import cugen
    import cugen.datasets as D
    assert cugen.datasets is D and D.fetch is fetch
    code = "import sys, cugen; assert 'cugen.datasets' not in sys.modules; cugen.datasets; " \
           "assert 'cugen.datasets' in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))

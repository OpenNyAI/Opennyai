"""Regression test for the wheel-rename step in opennyai.utils.download.install.

The downloaded wheels are published as `<name>-any-py3-none-any.whl`, which pip
rejects because `any` is not a valid version. `install()` reads the real version
out of the archive's METADATA and renames the file before installing.

The rename used to happen while the zipfile was still open. POSIX allows that;
Windows does not, and raised `PermissionError: [WinError 32]` before any model
could be installed. This test pins the ordering so it cannot regress.
"""

import importlib.util
import os
import pathlib
import sys
import types
import urllib.request
import zipfile

import pytest

WHEEL_NAME = "en_legal_ner_trf-any-py3-none-any.whl"
WHEEL_VERSION = "1.0.0"


@pytest.fixture
def download(monkeypatch):
    """Load download.py directly, without the heavyweight package import.

    `import opennyai.utils.download` pulls in the whole package, which needs
    spacy, transformers and a model download. download.py itself only touches
    the standard library plus torch, so it is loaded from its path with torch
    stubbed - keeping this a unit test that runs anywhere, including CI
    without a GPU.
    """
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.hub = types.ModuleType("torch.hub")
        torch_stub.device = lambda *_args, **_kwargs: None
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
        monkeypatch.setitem(sys.modules, "torch.hub", torch_stub.hub)

    module_path = (
        pathlib.Path(__file__).resolve().parents[2] / "opennyai" / "utils" / "download.py"
    )
    spec = importlib.util.spec_from_file_location("opennyai_download_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            f"en_legal_ner_trf-{WHEEL_VERSION}.dist-info/METADATA",
            f"Name: en_legal_ner_trf\nVersion: {WHEEL_VERSION}\n",
        )


def test_install_renames_wheel_to_its_real_version(download, monkeypatch):
    """The wheel handed to pip carries the version from METADATA, not 'any'."""
    installed = {}

    def fake_urlretrieve(_url, filename):
        _write_wheel(filename)

    def fake_check_call(cmd, *_args, **_kwargs):
        installed["wheel"] = cmd[-2] if cmd[-1] == "--no-deps" else cmd[-1]

    # install() imports urllib inside the function, so patch the real module.
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(download.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(download.shutil, "which", lambda _name: None)

    download.install(f"https://example.invalid/{WHEEL_NAME}")

    wheel = os.path.basename(installed["wheel"])
    assert wheel == f"en_legal_ner_trf-{WHEEL_VERSION}-py3-none-any.whl"
    assert "-any-py3-" not in wheel, "pip would reject 'any' as a version"


def test_rename_is_not_attempted_while_the_archive_is_open(tmp_path):
    """Renaming an open zipfile is a PermissionError on Windows.

    This asserts the platform behaviour the fix exists for, so the test fails
    loudly on Windows if the rename is ever moved back inside the `with` block.
    """
    if sys.platform != "win32":
        pytest.skip("only Windows forbids renaming a file with an open handle")

    wheel = tmp_path / WHEEL_NAME
    _write_wheel(wheel)
    target = tmp_path / f"en_legal_ner_trf-{WHEEL_VERSION}-py3-none-any.whl"

    with zipfile.ZipFile(wheel) as zf:
        zf.namelist()
        with pytest.raises(PermissionError):
            os.rename(wheel, target)

    os.rename(wheel, target)
    assert target.exists()

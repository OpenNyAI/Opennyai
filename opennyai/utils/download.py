import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

"""Functions for downloading opennyai ner models."""
PIP_INSTALLER_URLS = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
    "en_legal_ner_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl",
    "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "en_core_web_trf": "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
}
TORCH_PT_MODEL_URLS = {
    "RhetoricalRole": "https://huggingface.co/opennyaiorg/InRhetoricalRoles/resolve/main/InRhetoricalRoleModel.pt",
    "ExtractiveSummarizer": "https://huggingface.co/opennyaiorg/InExtractiveSummarizer/resolve/main/InExtractiveSummarizerModel.pt",
}
CACHE_DIR = os.path.join(str(Path.home()), '.opennyai')


def install(package: str):
    """
    It is used for installing pip wheel file for model supported
    Args:
        package (string): wheel file url
    """
    import tempfile
    import urllib.request

    # Download wheel to temp dir, renaming if it has an invalid version
    filename = package.rsplit("/", 1)[-1]
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, filename)
    urllib.request.urlretrieve(package, local_path)

    # Fix wheels with invalid version in filename by reading actual version from metadata
    if "-any-py3-" in filename:
        import zipfile
        with zipfile.ZipFile(local_path) as zf:
            for name in zf.namelist():
                if name.endswith("/METADATA"):
                    metadata = zf.read(name).decode()
                    for line in metadata.splitlines():
                        if line.startswith("Version:"):
                            version = line.split(":", 1)[1].strip()
                            fixed_filename = filename.replace("-any-py3-", f"-{version}-py3-")
                            fixed_path = os.path.join(tmp_dir, fixed_filename)
                            os.rename(local_path, fixed_path)
                            local_path = fixed_path
                            break
                    break

    uv_path = shutil.which("uv")
    env = os.environ.copy()
    try:
        if uv_path:
            subprocess.check_call(
                [uv_path, "pip", "install", local_path, "--no-deps"],
                stdout=subprocess.DEVNULL, env=env,
            )
        else:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", local_path, "--no-deps"],
                stdout=subprocess.DEVNULL,
            )
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)


def load_model_from_cache(model_name: str):
    """
    It is used for downloading model.pt files supported and developed by Opennyai
    Args:
        model_name (string): model name to download and save
    """
    if TORCH_PT_MODEL_URLS.get(model_name) is None:
        raise RuntimeError(f'{model_name} is not supported by opennyai, please check the name!')
    else:
        model_url = TORCH_PT_MODEL_URLS[model_name]
        os.makedirs(os.path.join(CACHE_DIR, model_name.lower()), exist_ok=True)
        return torch.hub.load_state_dict_from_url(model_url, model_dir=os.path.join(CACHE_DIR, model_name.lower()),
                                                  check_hash=True, map_location=torch.device('cpu'))

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

"""Functions for downloading opennyai ner models."""
PIP_INSTALLER_URLS = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
    "en_legal_ner_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl",
    "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.6.1/en_core_web_md-3.6.1-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.1/en_core_web_sm-3.6.1-py3-none-any.whl",
    "en_core_web_trf": "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.6.1/en_core_web_trf-3.6.1-py3-none-any.whl"}
TORCH_PT_MODEL_URLS = {
    "RhetoricalRole": "https://huggingface.co/opennyaiorg/InRhetoricalRoles/resolve/main/InRhetoricalRoleModel.pt",
    "ExtractiveSummarizer": "https://huggingface.co/opennyaiorg/InExtractiveSummarizer/resolve/main/InExtractiveSummarizerModel.pt"
}
CACHE_DIR = os.path.join(str(Path.home()), '.opennyai')


def install(package: str):
    """
    It is used for installing pip wheel file for model supported
    Args:
        package (string): wheel file url
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"], stdout=subprocess.DEVNULL
    )


def patch_model_spacy_version(model_name: str, version_range: str = ">=3.2.2,<4.0.0") -> None:
    """Broaden the spacy_version constraint in a model's meta.json so it loads on modern spacy.

    The frozen NER wheels shipped with spacy_version=">=3.2.2,<3.3.0", which causes
    spacy.load() to abort on spacy 3.4+.  This function overwrites that field after
    installation so the model is loadable without repackaging.

    Args:
        model_name: installed spacy model name (e.g. 'en_legal_ner_trf')
        version_range: new value to write into meta.json's spacy_version field
    """
    import spacy.util
    model_path = spacy.util.get_package_path(model_name)
    meta_path = model_path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["spacy_version"] = version_range
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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

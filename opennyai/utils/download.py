import os
import subprocess
import sys
import torch
from pathlib import Path

"""Functions for downloading opennyai ner models."""
models_url = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
    "en_legal_ner_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl",
    "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.2.0/en_core_web_md-3.2.0-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl",
    "en_core_web_trf": "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.2.0/en_core_web_trf-3.2.0-py3-none-any.whl",
    "RhetoricalRole": "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Model/model.pt"
}
CACHE_DIR = os.path.join(str(Path.home()), '.opennyai')


def install(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"], stdout=subprocess.DEVNULL
    )


def load_model_from_cache(model_name):
    if models_url.get(model_name) is None:
        raise RuntimeError(f'{model_name} is not supported by opennyai, please check the name!')
    else:
        model_url = models_url[model_name]
        os.makedirs(os.path.join(CACHE_DIR, model_name.lower()), exist_ok=True)
        return torch.hub.load_state_dict_from_url(model_url, model_dir=os.path.join(CACHE_DIR, model_name.lower()),
                                                  check_hash=True, map_location=torch.device('cpu'))

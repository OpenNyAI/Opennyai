"""Functions for downloading opennyai ner models."""
models_url = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
    "en_legal_ner_sm": "https://huggingface.co/opennyaiorg/en_legal_ner_sm/resolve/main/en_legal_ner_sm-any-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl"
}


def install(package):
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"],stdout=subprocess.DEVNULL
    )

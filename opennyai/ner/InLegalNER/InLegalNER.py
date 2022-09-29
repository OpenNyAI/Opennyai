import spacy
from wasabi import msg
from .entity_recognizer_utils import extract_entities_from_judgment_text
from .download import install, models_url


class InLegalNER:
    def __init__(self, model_name='en_legal_ner_trf'):
        for mdl in [model_name, 'en_core_web_sm']:
            if mdl not in spacy.util.get_installed_models():
                msg.info(f'Installing {mdl} this is a one time process!!')
                if models_url.get(mdl) is not None:
                    install(models_url[mdl])
                else:
                    raise RuntimeError(f'{model_name} doesn\'t exist in list of available opennyai ner models')
        try:
            if spacy.prefer_gpu():
                msg.info(title='GPU')
            else:
                msg.info(title='CPU')
            spacy.prefer_gpu()
        except:
            msg.info(title='CPU')
        self.model_name = model_name
        self.nlp = spacy.load(self.model_name)
        try:
            self.__splitter_nlp__ = spacy.load('en_core_web_sm')
        except:
            raise RuntimeError(
                'There was an error while loading en_core_web_sm\n To rectify try running:\n pip install -U https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl')

    def __call__(self, text, do_sentence_level=True):
        nlp_doc = extract_entities_from_judgment_text(txt=text, legal_nlp=self.nlp,
                                                   preamble_splitting_nlp=self.__splitter_nlp__,
                                                   do_sentence_level=do_sentence_level)
        return nlp_doc

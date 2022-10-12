import spacy
from wasabi import msg
from .entity_recognizer_utils import extract_entities_from_judgment_text
from .postprocessing_utils import precedent_coref_resol, other_person_coref_res, pro_statute_coref_resol, \
    remove_overlapping_entities
from opennyai.utils.download import install, models_url


class InLegalNER:
    def __init__(self, model_name='en_legal_ner_trf', sentence_splitter_model_name='en_core_web_trf'):
        for mdl in [model_name, sentence_splitter_model_name]:
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
            self.__splitter_nlp__ = spacy.load(sentence_splitter_model_name,
                                               exclude=['attribute_ruler', 'lemmatizer', 'ner'])
        except:
            raise RuntimeError(
                f'There was an error while loading en_core_web_sm\n To rectify try running:\n pip install -U {models_url[sentence_splitter_model_name]}')

    def __call__(self, text, do_sentence_level=True, do_postprocess=False):
        nlp_doc = extract_entities_from_judgment_text(txt=text, legal_nlp=self.nlp,
                                                      preamble_splitting_nlp=self.__splitter_nlp__,
                                                      do_sentence_level=do_sentence_level)
        if do_sentence_level and do_postprocess:
            precedent_clusters = precedent_coref_resol(nlp_doc)

            other_person_entites = other_person_coref_res(nlp_doc)
            pro_sta_clusters = pro_statute_coref_resol(nlp_doc)

            all_entities = remove_overlapping_entities(nlp_doc.ents, pro_sta_clusters)

            all_entities.extend(other_person_entites)

            nlp_doc.ents = all_entities
            nlp_doc.set_extension("precedent_clusters", default=precedent_clusters, force=True)
            nlp_doc.set_extension("provision_statute_clusters", default=pro_sta_clusters, force=True)

        return nlp_doc

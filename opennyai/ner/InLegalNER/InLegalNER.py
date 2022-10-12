import spacy
from wasabi import msg
from tqdm import tqdm
from .entity_recognizer_utils import extract_entities_from_judgment_text
from .postprocessing_utils import precedent_coref_resol, other_person_coref_res, pro_statute_coref_resol, \
    remove_overlapping_entities
from opennyai.utils.download import install, models_url


class InLegalNER:
    def __init__(self, model_name='en_legal_ner_trf'):
        if model_name not in spacy.util.get_installed_models():
            msg.info(f'Installing {model_name} this is a one time process!!')
            if models_url.get(model_name) is not None:
                install(models_url[model_name])
            else:
                raise RuntimeError(f'{model_name} doesn\'t exist in list of available opennyai ner models')
        try:
            if spacy.prefer_gpu():
                msg.info(title='NER will run on GPU')
            else:
                msg.info(title='NER will run on CPU')
            spacy.prefer_gpu()
        except:
            msg.info(title='NER will run on CPU')
        self.model_name = model_name
        self.nlp = spacy.load(self.model_name)

    def __call__(self, data, do_sentence_level=True, do_postprocess=True, mini_batch_size=40000, verbose=False):
        processed_data = []
        if verbose:
            msg.info('Processing documents')
        for to_process in tqdm(data, disable=not verbose):
            nlp_doc = extract_entities_from_judgment_text(to_process=to_process, legal_nlp=self.nlp,
                                                          do_sentence_level=do_sentence_level,
                                                          mini_batch_size=mini_batch_size)
            if do_sentence_level and do_postprocess:
                precedent_clusters = precedent_coref_resol(nlp_doc)

                other_person_entites = other_person_coref_res(nlp_doc)
                pro_sta_clusters = pro_statute_coref_resol(nlp_doc)

                all_entities = remove_overlapping_entities(nlp_doc.ents, pro_sta_clusters)

                all_entities.extend(other_person_entites)

                nlp_doc.ents = all_entities
                nlp_doc.set_extension("precedent_clusters", default=precedent_clusters, force=True)
                nlp_doc.set_extension("provision_statute_clusters", default=pro_sta_clusters, force=True)
            processed_data.append(nlp_doc)
        if len(processed_data) == 1:
            return processed_data[0]
        else:
            return processed_data

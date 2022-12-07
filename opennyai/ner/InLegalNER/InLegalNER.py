import copy

import spacy
from tqdm import tqdm
from wasabi import msg

from opennyai.utils.download import install, PIP_INSTALLER_URLS
from .entity_recognizer_utils import extract_entities_from_judgment_text
from .postprocessing_utils import precedent_coref_resol, other_person_coref_res, pro_statute_coref_resol, \
    remove_overlapping_entities


class InLegalNER:
    def __init__(self, model_name='en_legal_ner_trf', use_gpu=True):
        """Returns object of InLegalNER class.
         It is used for loading InLegalNER model in memory.
        Args:
            model_name (string): Accepts a model name of spacy as InLegalNER that will be used for NER inference
            available models are 'en_legal_ner_trf', 'en_legal_ner_sm'
            use_gpu (bool): Functionality to give a choice whether to use GPU for inference or not
             Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in
             documentation
        """
        if model_name not in spacy.util.get_installed_models():
            msg.info(f'Installing {model_name}. This is a one time process!!')
            if PIP_INSTALLER_URLS.get(model_name) is not None:
                install(PIP_INSTALLER_URLS[model_name])
            else:
                raise RuntimeError(f'{model_name} doesn\'t exist in list of available opennyai ner models')
        if use_gpu:
            try:
                if spacy.prefer_gpu():
                    msg.info(title='NER will run on GPU!')
                else:
                    msg.info(title='NER will run on CPU!')
                spacy.prefer_gpu()
            except:
                msg.info(title='NER will run on CPU!')
        else:
            msg.info(title='NER will run on CPU!')
        self.model_name = model_name
        self.nlp = spacy.load(self.model_name)

    def __call__(self, data, do_sentence_level=True, do_postprocess=True, mini_batch_size=40000, verbose=False,
                 statute_shortforms_path=''):
        """Returns doc of InLegalNER nlp.
         It is used for doing inference on input doc using InLegalNER model in memory.
        Args:
            data (object of Data class): Data class object containing input docs
            do_sentence_level (bool): To perform inference at sentence level or not, at sentence level it better accuracy
            do_postprocess (bool): To perform post-processing over processed doc
            mini_batch_size (int): This accepts an int as batch size for processing of a document,
            if length of document is bigger that given batch size it will be chunked and then processed.
            verbose (bool): Set it to if you want to see progress bar while processing happens
        Note:
            do_postprocess is depends on do_sentence_level so for doing postprocessing doing sentence level inference
            is mandatory.
        """
        processed_data = []
        if verbose:
            msg.info('Processing documents with Legal NER!!!')
        for to_process in tqdm(data, disable=not verbose):
            nlp_doc = extract_entities_from_judgment_text(to_process=to_process, legal_nlp=self.nlp,
                                                          do_sentence_level=do_sentence_level,
                                                          mini_batch_size=mini_batch_size)
            nlp_doc.user_data['doc_id'] = to_process['file_id']
            nlp_doc.user_data['original_text'] = to_process['original_text']
            nlp_doc.user_data['preamble_end_char_offset'] = len(to_process['preamble_doc'].text)
            if do_sentence_level and do_postprocess:
                all_entities = None
                postprocessing_success = True
                precedent_clusters = precedent_coref_resol(nlp_doc)

                other_person_entites = other_person_coref_res(nlp_doc)

                pro_sta_clusters, stat_clusters = pro_statute_coref_resol(nlp_doc, statute_shortforms_path)
                if pro_sta_clusters:
                    all_entities = remove_overlapping_entities(nlp_doc.ents, pro_sta_clusters)

                    all_entities.extend(other_person_entites)
                elif pro_sta_clusters is None and stat_clusters is None:
                    postprocessing_success = False

                if all_entities:
                    nlp_doc.ents = all_entities
                else:
                    ents = list(nlp_doc.ents)
                    ents.extend(other_person_entites)
                    ents = spacy.util.filter_spans(tuple(ents))
                    nlp_doc.ents = ents

                if precedent_clusters:
                    nlp_doc.user_data['precedent_clusters'] = precedent_clusters
                elif precedent_clusters is None:
                    postprocessing_success = False

                if pro_sta_clusters:
                    nlp_doc.user_data['provision_statute_pairs'] = pro_sta_clusters
                elif pro_sta_clusters is None:
                    postprocessing_success = False

                if stat_clusters:
                    nlp_doc.user_data['statute_clusters'] = stat_clusters
                elif stat_clusters is None:
                    postprocessing_success = False

                if not postprocessing_success:
                    msg.warn(
                        f'''There was some issue while performing postprocessing for doc id {to_process['file_id']}. 
                        Some of postprocessing info may be absent because of this in doc.''')
            processed_data.append(nlp_doc)
        if len(processed_data) == 1:
            return processed_data[0]
        else:
            return processed_data

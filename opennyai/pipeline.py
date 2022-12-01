import copy

from wasabi import msg

import opennyai.ner as InLegalNER
from opennyai.rhetorical_roles.rhetorical_roles import RhetoricalRolePredictor
from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer


class Pipeline:
    def __init__(self, components=None, use_gpu=True, verbose=True, ner_model_name='en_legal_ner_trf',
                 ner_mini_batch_size=40000, ner_do_sentence_level=True, ner_do_postprocess=True,
                 ner_statute_shortforms_path='', summarizer_summary_length=0.0):
        self.__ner_mini_batch_size__ = ner_mini_batch_size
        self.__do_sentence_level__ = ner_do_sentence_level
        self.__do_postprocess__ = ner_do_postprocess
        self.__statute_shortforms_path__ = ner_statute_shortforms_path
        default_component_values = ['NER', 'Rhetorical_Role', 'Summarizer']
        if components is None:
            components = default_component_values
        if 'Summarizer' in components:
            if 'Rhetorical_Role' not in components:
                components.append('Rhetorical_Role')
        assert all(
            [i in default_component_values for i in components]), f"Invalid component value given in {components}"
        self.components = components
        self.__verbose__ = verbose
        if 'NER' in components:
            if self.__verbose__:
                msg.info('Loading NER...')
            self.__ner_extractor__ = InLegalNER.load(use_gpu=use_gpu, model_name=ner_model_name)

        if 'Rhetorical_Role' in components or 'Summarizer' in components:
            if self.__verbose__:
                msg.info('Loading Rhetorical Role...')
            self.__rr_model__ = RhetoricalRolePredictor(use_gpu=use_gpu, verbose=verbose)

        if 'Summarizer' in components:
            if self.__verbose__:
                msg.info('Loading Extractive summarizer...')
            self.__summarizer__ = ExtractiveSummarizer(use_gpu=use_gpu, verbose=verbose,
                                                       summary_length=summarizer_summary_length)

    @staticmethod
    def __combine_model_outputs__(ner_json_results=None, rr_output=None, summary_output=None):
        '''combines the outputs of 3 models into single list'''
        combined_results = {}
        ####### Add NER results
        if ner_json_results:
            for doc_ner in ner_json_results:
                doc_id = doc_ner['id'].split('_')[1]
                if combined_results.get(doc_id) is None:
                    combined_results[doc_id] = {'id': doc_id, 'data': {'text': doc_ner['data']['text'],
                                                                       'preamble_end_char_offset': doc_ner['data'].get(
                                                                           'preamble_end_char_offset')}}

                combined_results[doc_id]['annotations'] = doc_ner['annotations']

        ####### Add RR results
        if rr_output:
            for doc_rr in rr_output:
                doc_id = doc_rr['id'].split('_')[1]
                if combined_results.get(doc_id) is None:
                    combined_results[doc_id] = {'id': doc_id,
                                                'data': {'text': doc_rr['data']['text'],
                                                         'preamble_end_char_offset': doc_rr['data'].get(
                                                             'preamble_end_char_offset')}}

                combined_results[doc_id]['annotations'] = doc_rr['annotations']

        ####### Add summary results
        if summary_output:
            for doc_summary in summary_output:
                doc_id = doc_summary['id'].split('_')[1]
                combined_results[doc_id]['summary'] = doc_summary['summaries']

        if ner_json_results and rr_output:
            for doc_ner in ner_json_results:
                entities = [entity for annotation in doc_ner['annotations'] for entity in
                            annotation['entities']]
                doc_id = doc_ner['id'].split('_')[1]
                for entity in entities:
                    for sent in combined_results[doc_id]['annotations']:
                        if sent.get('entities') is None:
                            sent['entities'] = []
                        if entity['start'] >= sent['start'] and entity['end'] <= sent['end']:
                            sent['entities'].append(entity)
                            break
                combined_results[doc_id]['annotations'] = copy.deepcopy(combined_results[doc_id]['annotations'])

        return [result for doc_id, result in combined_results.items()]

    @staticmethod
    def __postprocess_ner_to_sentence_level__(doc):
        id = "LegalNER_" + doc.user_data['doc_id']
        final_output = InLegalNER.get_json_from_spacy_doc(doc)
        output = {'id': id, 'annotations': [],
                  'data': {'text': doc.text, 'original_text': doc.user_data['original_text'],
                           'preamble_end_char_offset': doc.user_data['preamble_end_char_offset']}}
        for sent in doc.sents:
            import uuid
            uid = uuid.uuid4()
            id = uid.hex
            temp = copy.deepcopy({"id": id,
                                  "start": sent.start_char,
                                  "end": sent.end_char,
                                  "text": sent.text,
                                  "entities": []})
            for entity in final_output['annotations']:
                if entity['start'] >= temp['start'] and entity['end'] <= temp['end']:
                    temp['entities'].append(entity)
            output['annotations'].append(temp)
        return output

    def __call__(self, data):
        ner_json_results, self._ner_model_output, self._rr_model_output, self._summarizer_model_output = None, None, None, None
        if 'NER' in self.components:
            self._ner_model_output = self.__ner_extractor__(data, verbose=self.__verbose__,
                                                            do_sentence_level=self.__do_sentence_level__,
                                                            do_postprocess=self.__do_postprocess__,
                                                            mini_batch_size=self.__ner_mini_batch_size__,
                                                            statute_shortforms_path=self.__statute_shortforms_path__)
            if not isinstance(self._ner_model_output, list):
                self._ner_model_output = [self._ner_model_output]
            ner_json_results = [self.__postprocess_ner_to_sentence_level__(doc) for doc in self._ner_model_output]

        if 'Rhetorical_Role' in self.components or 'Summarizer' in self.components:
            self._rr_model_output = self.__rr_model__(data)
        if 'Summarizer' in self.components:
            self._summarizer_model_output = self.__summarizer__(self._rr_model_output)

        return self.__combine_model_outputs__(ner_json_results, self._rr_model_output, self._summarizer_model_output)

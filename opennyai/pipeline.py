import opennyai.ner as InLegalNER
from opennyai import RhetoricalRolePredictor
from opennyai import ExtractiveSummarizer


class Pipeline:
    def __init__(self, components=['NER', 'Rhetorical_Role', 'Summarizer'], use_gpu=True, verbose=False):
        self.components = components
        self.__verbose__ = verbose
        if 'NER' in components:
            self._ner_extractor = InLegalNER.load(use_gpu=use_gpu)

        if 'Rhetorical_Role' in components or 'Summarizer' in components:
            self._rr = RhetoricalRolePredictor(use_gpu=use_gpu,verbose=verbose)

        if 'Summarizer' in components:
            self._summarizer = ExtractiveSummarizer(use_gpu=use_gpu,verbose=verbose)

    def combine_model_outputs(self, ner_json_results=None, rr_output=None, summary_output=None):
        '''combines the outputs of 3 models into single list'''
        combined_results = {}
        ####### Add NER results
        if ner_json_results:
            for doc_ner in ner_json_results:
                doc_id = doc_ner['id'].split('_')[1]
                if combined_results.get(doc_id) is None:
                    combined_results[doc_id] = {'id': doc_id, 'data': {'text': doc_ner['data']['text']}}

                combined_results[doc_id]['ner_annotations'] = doc_ner['annotations']

        ####### Add RR results
        if rr_output:
            for doc_rr in rr_output:
                doc_id = doc_rr['id'].split('_')[1]
                if combined_results.get(doc_id) is None:
                    combined_results[doc_id] = {'id': doc_id, 'data': {'text': doc_rr['data']['text']}}

                combined_results[doc_id]['rr_annotations'] = doc_rr['annotations']

        ####### Add summary results
        if summary_output:
            for doc_summary in summary_output:
                doc_id = doc_summary['id'].split('_')[1]
                combined_results[doc_id]['summary'] = doc_summary['summaries']

        return [result for doc_id,result in combined_results.items()]

    def __call__(self, data):
        ner_json_results, rr_output, summary_output = None, None, None
        if 'NER' in self.components:
            ner_results = self._ner_extractor(data,verbose=self.__verbose__)
            ner_json_results = [InLegalNER.get_json_from_spacy_doc(i) for i in ner_results]

        if 'Rhetorical_Role' in self.components or 'Summarizer' in self.components:
            rr_output = self._rr(data)

        if 'Summarizer' in self.components:
            summary_output = self._summarizer(rr_output)

        return self.combine_model_outputs(ner_json_results, rr_output, summary_output)

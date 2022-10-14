from opennyai.ner.InLegalNER.InLegalNER import InLegalNER
import copy
from hashlib import sha256


def load(model_name: str = 'en_legal_ner_trf', use_gpu: bool = True):
    """Returns object of InLegalNER class.
     It is used for loading InLegalNER model in memory.
    Args:
        model_name (string): Accepts a model name of spacy as InLegalNER that will be used for NER inference
        available models are 'en_legal_ner_trf', 'en_legal_ner_sm'
        use_gpu (bool): Functionality to give a choice whether to use GPU for inference or not
         Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in
         documentation
    """
    AVAILABLE_LEGAL_NER_MODELS = ['en_legal_ner_trf', 'en_legal_ner_sm']
    if model_name not in AVAILABLE_LEGAL_NER_MODELS:
        raise RuntimeError(f'{model_name} doesn\'t exit in list of available models {AVAILABLE_LEGAL_NER_MODELS}')
    return InLegalNER(model_name, use_gpu)


def find_parent_child_id(cluster: list, ls_formatted_doc: dict):
    cluster = [i for i in cluster if type(i) != str]
    entities_list = ls_formatted_doc['annotations'][0]['result']
    parent_id = ''
    child_ids = []
    parent = cluster[0]
    childs = cluster[1:]
    for i in entities_list:
        if parent_id == '' and i['value']['start'] == parent.start_char and i['value']['end'] == parent.end_char:
            parent_id = i['id']
        for child in childs:
            if i['value']['start'] == child.start_char and i['value']['end'] == child.end_char:
                child_ids.append(i['id'])
    return parent_id, child_ids


def update_json_with_clusters(ls_formatted_doc: dict, precedent_clusters: list, provision_statute_clusters: list):
    entities_list = copy.deepcopy(ls_formatted_doc['annotations'][0]['result'])
    provision_statute_clusters_ids = [find_parent_child_id(i, ls_formatted_doc) for i in provision_statute_clusters]
    precedent_clusters_ids = [find_parent_child_id(precedent_clusters[i], ls_formatted_doc) for i in
                              precedent_clusters.keys() if len(precedent_clusters[i]) > 1]
    for i in precedent_clusters_ids:
        parent_id = i[0]
        parent_text = ''
        for _ in entities_list:
            if _['id'] == parent_id:
                parent_text = _['value']['text']
        for j in i[1]:
            for _ in ls_formatted_doc['annotations'][0]['result']:
                if _['id'] == j:
                    _['meta']['text'].append(parent_text)
    for i in provision_statute_clusters_ids:
        parent_id = i[0]
        child_texts = []
        for j in i[1]:
            for _ in entities_list:
                if _['id'] == j:
                    child_texts.append(_['value']['text'])
        for _ in ls_formatted_doc['annotations'][0]['result']:
            if _['id'] == parent_id:
                _['meta']['text'] = child_texts

    return ls_formatted_doc


def get_json_from_spacy_doc(doc) -> dict:
    """Returns dict of InLegalNER doc.
    Args:
        doc: InLegalNER doc
    """
    id = "LegalNER_" + doc.user_data['doc_id']
    output = {'id': id, 'annotations': [{'result': []}],
              'data': {'text': doc.text, 'original_text': doc.user_data['original_text']}}
    for ent in doc.ents:
        import uuid
        uid = uuid.uuid4()
        id = uid.hex
        output['annotations'][0]['result'].append(copy.deepcopy({"id": id, "meta": {"text": []},
                                                                 "type": "labels",
                                                                 "value": {
                                                                     "start": ent.start_char,
                                                                     "end": ent.end_char,
                                                                     "text": ent.text,
                                                                     "labels": [ent.label_],
                                                                 }, "to_name": "text",
                                                                 "from_name": "label"
                                                                 }))

    final_output = update_json_with_clusters(copy.deepcopy(output), doc.user_data['precedent_clusters'],
                                             doc.user_data['provision_statute_clusters'])

    return final_output


ner_displacy_option = {
    "colors": {"PETITIONER": "yellow", "RESPONDENT": "green", "JUDGE": "pink", "WITNESS": "purple", "LAWYER": "red",
               "OTHER_PERSON": "cyan",
               "PETITIONER_match": "yellow", "RESPONDENT_match": "green", "JUDGE_match": "pink",
               "WITNESS_match": "purple", "LAWYER_match": "red",
               "PROVISION": "#33E9FF", "STATUTE": "#1C4D53", "GPE": "#A6A82F", "ORG": "#603255", "COURT": "#56A065",
               "DATE": "#804538", "CASE_NUMBER": "#71326E"}}

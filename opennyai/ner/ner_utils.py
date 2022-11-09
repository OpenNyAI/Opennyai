import collections
import copy

import pandas as pd

from opennyai.ner.InLegalNER.InLegalNER import InLegalNER


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


def update_json_with_clusters(ls_formatted_doc: dict, precedent_clusters: dict, provision_statute_clusters: list,
                              statute_clusters: dict):
    for entity, statute, normalized_provision, normalized_statute in provision_statute_clusters:
        for result in ls_formatted_doc['annotations']:
            if result['start'] == entity.start_char and result['end'] == entity.end_char:
                result['normalized_name'] = str(normalized_provision) + ' of ' + str(normalized_statute)

    for statute_head in statute_clusters.keys():
        for entity in statute_clusters[statute_head]:
            for result in ls_formatted_doc['annotations']:
                if result['start'] == entity.start_char and result['end'] == entity.end_char:
                    result['normalized_name'] = str(statute_head)

    for precedent_head in precedent_clusters.keys():
        for entity in precedent_clusters[precedent_head]:
            for result in ls_formatted_doc['annotations']:
                if result['start'] == entity.start_char and result['end'] == entity.end_char:
                    result['normalized_name'] = str(precedent_head)

    return ls_formatted_doc


def get_json_from_spacy_doc(doc) -> dict:
    """Returns dict of InLegalNER doc.
    Args:
        doc: InLegalNER doc
    """
    id = "LegalNER_" + doc.user_data['doc_id']
    output = {'id': id, 'annotations': [],
              'data': {'text': doc.text, 'original_text': doc.user_data['original_text']}}
    for ent in doc.ents:
        import uuid
        uid = uuid.uuid4()
        id = uid.hex
        output['annotations'].append(copy.deepcopy({"id": id,
                                                    "normalized_name": ent.text,
                                                    "start": ent.start_char,
                                                    "end": ent.end_char,
                                                    "text": ent.text,
                                                    "labels": [ent.label_]}))

    if doc.user_data.get('precedent_clusters') is not None and doc.user_data.get(
            'provision_statute_pairs') is not None and doc.user_data.get('statute_clusters') is not None:
        final_output = update_json_with_clusters(copy.deepcopy(output), doc.user_data['precedent_clusters'],

                                                 doc.user_data['provision_statute_pairs'],
                                                 doc.user_data['statute_clusters'])

        return final_output
    else:
        return output


def get_csv(doc, f_name, save_path):
    df = pd.DataFrame(columns=['file_name', 'entity', 'label', 'normalised_entities'])
    file_name = []
    entity = []
    label = []
    normalised_entities = []

    for pro_ent in doc.user_data['provision_statute_pairs']:
        file_name.append(f_name)
        entity.append(pro_ent[0])
        label.append('PROVISION')
        normalised_entities.append(pro_ent[2] + ' of ' + pro_ent[3])
    for pre_head in doc.user_data['precedent_clusters'].keys():

        for ent in doc.user_data['precedent_clusters'][pre_head]:
            file_name.append(f_name)
            entity.append(ent)
            label.append('PRECEDENT')
            normalised_entities.append(pre_head)
    for pre_head in doc.user_data['statute_clusters'].keys():

        for ent in doc.user_data['statute_clusters'][pre_head]:
            file_name.append(f_name)
            entity.append(ent)
            label.append('STATUTE')
            normalised_entities.append(pre_head)

    for ent in doc.ents:
        if ent not in entity:
            file_name.append(f_name)
            entity.append(ent)
            label.append(ent.label_)
            normalised_entities.append('')
    entity_text = [ent.text for ent in entity]
    df['file_name'] = file_name
    df['entity'] = entity_text
    df['label'] = label
    df['normalised_entities'] = normalised_entities

    df.to_csv(save_path)


def get_unique_precedent_count(doc):
    new_clusters = {}
    clusters = doc.user_data['precedent_clusters']
    for c in clusters.keys():
        new_clusters[c] = len(clusters[c])

    return new_clusters


def get_unique_provision_count(doc):
    clusters = doc.user_data['provision_statute_pairs']
    provisions = [cluster[2] + ' of ' + cluster[3] for cluster in clusters]
    frequency = dict(collections.Counter(provisions))

    return frequency


def get_unique_statute_count(doc):
    clusters = doc.user_data['provision_statute_pairs']
    statutes = [cluster[3] for cluster in clusters]
    frequency = dict(collections.Counter(statutes))

    return frequency


ner_displacy_option = {
    'colors': {'COURT': "#bbabf2", 'PETITIONER': "#f570ea", "RESPONDENT": "#cdee81", 'JUDGE': "#fdd8a5",
               "LAWYER": "#f9d380", 'WITNESS': "violet", "STATUTE": "#faea99", "PROVISION": "yellow",
               'CASE_NUMBER': "#fbb1cf", "PRECEDENT": "#fad6d6", 'DATE': "#b1ecf7", 'OTHER_PERSON': "#b0f6a2",
               'ORG': '#a57db5', 'GPE': '#7fdbd4'}}

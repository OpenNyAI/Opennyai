from opennyai.ner.InLegalNER.InLegalNER import InLegalNER


def load(model_name='en_legal_ner_trf'):
    AVAILABLE_MODELS = ['en_legal_ner_trf', 'en_legal_ner_sm']
    if model_name not in AVAILABLE_MODELS:
        raise RuntimeError(f'{model_name} doesn\'t exit in list of available models {AVAILABLE_MODELS}')
    return InLegalNER(model_name)


def get_json_from_spacy_doc(doc):
    import uuid, copy
    uid = uuid.uuid4()
    id = "LegalNER_" + str(uid.hex)
    output = {'id': id, 'annotations': [{'result': []}], 'data': {'text': doc.text}}
    for ent in doc.ents:
        import uuid
        uid = uuid.uuid4()
        output['annotations'][0]['result'].append(copy.deepcopy({
            "value": {
                "start": ent.start_char,
                "end": ent.end_char,
                "text": ent.text,
                "labels": [ent.label_],
                "id": uid.hex
            }
        }))
    return output


ner_displacy_option = {
    "colors": {"PETITIONER": "yellow", "RESPONDENT": "green", "JUDGE": "pink", "WITNESS": "purple", "LAWYER": "red",
               "OTHER_PERSON": "cyan",
               "PETITIONER_match": "yellow", "RESPONDENT_match": "green", "JUDGE_match": "pink",
               "WITNESS_match": "purple", "LAWYER_match": "red",
               "PROVISION": "#33E9FF", "STATUTE": "#1C4D53", "GPE": "#A6A82F", "ORG": "#603255", "COURT": "#56A065",
               "DATE": "#804538", "CASE_NUMBER": "#71326E"}}

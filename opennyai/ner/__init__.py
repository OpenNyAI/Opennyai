def load(model_name='en_legal_ner_trf'):
    from opennyai.ner.InLegalNER.InLegalNER import InLegalNER
    AVAILABLE_MODELS = ['en_legal_ner_trf', 'en_legal_ner_sm']
    if model_name not in AVAILABLE_MODELS:
        ValueError(f'{model_name} doesn\'t exit in list of available models {AVAILABLE_MODELS}')
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

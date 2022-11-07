import json
import os

from opennyai import Pipeline
from opennyai.ner import get_json_from_spacy_doc
from opennyai.utils import Data
from ..utils import reset_ids, reset_sent_scores

# Execute pipeline on the text
saved_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text1 = open(os.path.join(saved_data_path, 'examples/judgement_texts/72703592.txt')).read()
text2 = open(os.path.join(saved_data_path, 'examples/judgement_texts/811682.txt')).read()
texts_to_process = [text1, text2]
data = Data(texts_to_process)
pipeline = Pipeline(components=['NER', 'Rhetorical_Role', 'Summarizer'], use_gpu=True)
results = pipeline(data)

# load saved results
ner_model_output_text1 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/ner_output_files/72703592.json'))))
rr_model_output_text1 = reset_sent_scores(reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/rr_output_files/72703592.json')))))
summarizer_model_output_text1 = json.load(
    open(os.path.join(saved_data_path, 'examples/summarizer_output_files/72703592.json')))

ner_model_output_text2 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/ner_output_files/811682.json'))))
rr_model_output_text2 = reset_sent_scores(reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/rr_output_files/811682.json')))))
summarizer_model_output_text2 = json.load(
    open(os.path.join(saved_data_path, 'examples/summarizer_output_files/811682.json')))

pipeline_output_text1 = reset_sent_scores(reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/pipeline_output_files/72703592.json')))))
pipeline_output_text2 = reset_sent_scores(reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/pipeline_output_files/811682.json')))))


def test_ner_output():
    ner_output_valid = True
    output1 = reset_ids(get_json_from_spacy_doc(pipeline._ner_model_output[0]))
    output2 = reset_ids(get_json_from_spacy_doc(pipeline._ner_model_output[1]))
    if output1 != ner_model_output_text1 or output2 != ner_model_output_text2:
        ner_output_valid = False
    assert ner_output_valid


def test_rr_output():
    rr_output_valid = True
    output1 = reset_sent_scores(reset_ids(pipeline._rr_model_output[0]))
    output2 = reset_sent_scores(reset_ids(pipeline._rr_model_output[1]))
    if output1 != rr_model_output_text1 or output2 != rr_model_output_text2:
        rr_output_valid = False
    assert rr_output_valid


def test_summarizer_output():
    summarizer_output_valid = True
    output1 = pipeline._summarizer_model_output[0]
    output2 = pipeline._summarizer_model_output[1]
    if output1 != summarizer_model_output_text1 or output2 != summarizer_model_output_text2:
        summarizer_output_valid = False
    assert summarizer_output_valid


def test_pipeline_output():
    pipeline_output_valid = True
    output1 = reset_sent_scores(reset_ids(results[0]))
    output2 = reset_sent_scores(reset_ids(results[1]))
    if output1 != pipeline_output_text1 or output2 != pipeline_output_text2:
        pipeline_output_valid = False
    assert pipeline_output_valid

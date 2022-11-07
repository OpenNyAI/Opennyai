import json
import os

from opennyai import Pipeline
from opennyai.ner import get_json_from_spacy_doc
from opennyai.utils import Data
from ..utils import reset_ids

# Execute pipeline on the text
saved_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text1 = open(os.path.join(saved_data_path, 'examples/judgement_texts/72703592.txt')).read()
text2 = open(os.path.join(saved_data_path, 'examples/judgement_texts/811682.txt')).read()
texts_to_process = [text1, text2]
data = Data(texts_to_process)
pipeline = Pipeline(components=['NER', 'Rhetorical_Role', 'Summarizer'], use_gpu=False)
results = pipeline(data)

# load saved results
ner_model_output_text1 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/ner_output_files/72703592.json'))))
rr_model_output_text1 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/rr_output_files/72703592.json'))))
summarizer_model_output_text1 = json.load(
    open(os.path.join(saved_data_path, 'examples/summarizer_output_files/72703592.json')))

ner_model_output_text2 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/ner_output_files/811682.json'))))
rr_model_output_text2 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/rr_output_files/811682.json'))))
summarizer_model_output_text2 = json.load(
    open(os.path.join(saved_data_path, 'examples/summarizer_output_files/811682.json')))

pipeline_output_text1 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/pipeline_output_files/72703592.json'))))
pipeline_output_text2 = reset_ids(
    json.load(open(os.path.join(saved_data_path, 'examples/pipeline_output_files/811682.json'))))


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

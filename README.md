<a href="https://github.com/OpenNyAI/Opennyai"><img src="https://github.com/OpenNyAI/Opennyai/raw/master/asset/final-logo-01.jpeg" width="190" height="65" align="right" /></a>

# Opennyai : An efficient NLP Pipeline for Indian Legal documents

[![PyPI version](https://badge.fury.io/py/opennyai.svg)](https://pypi.org/project/opennyai/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/OpenNyAI/Opennyai)

Opennyai is a natural language preprocessing framework made with SpaCy. Its pipeline has achieved State-of-the-Art
performance on Named entity recognition on Indian legal NER.
Try [demo](https://huggingface.co/opennyaiorg/en_legal_ner_trf)

We are giving access to three model developed by us which specializes in Indian legal domain:

* NER
* Rhetorical Role prediction **Coming Soon**
* Extractive Summarizer

# 🔧 Installation

To get started using opennyai simply install it using pip by running the following line in your terminal:

```bash
pip install opennyai
```

Note: if you want to utilize spacy with GPU please install [Cupy](https://anaconda.org/conda-forge/cupy)
/[cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) dependency of appropriate version. For spacy installation with
cupy refer to [page](https://spacy.io/usage)

Remember you need spacy of 3.2.4 version for models to work perfectly.

# 👩‍💻 Usage NER

To use the NER model you first have to select and load model from given list.

* en_legal_ner_trf (This model provides the highest accuracy)
* en_legal_ner_sm (This model provides the highest efficiency)

Available preprocessing models are ['en_core_web_md', 'en_core_web_sm', 'en_core_web_trf']

To download and load a model simply execute:

```python
import opennyai.ner as InLegalNER
from opennyai.utils import Data

text = 'Section 319 Cr.P.C. contemplates a situation where the evidence adduced by the prosecution for Respondent No.3-G. Sambiah on 20th June 1984'
# you can pass multiple documents in form of list to below line of code
data = Data(text, preprocessing_nlp_model='en_core_web_trf', mini_batch_size=40000, use_gpu=True, use_cache=True,
            verbose=False)
nlp = InLegalNER.load('en_legal_ner_trf', use_gpu=True)
doc = nlp(data, do_sentence_level=True,
          do_postprocess=True, mini_batch_size=40000,
          verbose=False)  # set do_sentence_level and do_postprocess to False if you pass a sentence 
identified_entites = [(ent, ent.label_) for ent in doc.ents]
```

Result:

```
[(Section 319, 'PROVISION'),
 (Cr.P.C., 'STATUTE'),
 (G. Sambiah, 'RESPONDENT'),
 (20th June 1984, 'DATE')]
 ```

To get result in json format with span information:

```python
json_result = InLegalNER.get_json_from_spacy_doc(doc)
```

Note: You can import generated json to label studio and visualize all the details

# 👩‍💻 Usage Extractive Summarizer

To use the Extractive Summarizer model you first need to have rhetorical role module output.

Here we will use a pre-saved for demo purpose

To use model simply execute:

```python
import json
from opennyai import ExtractiveSummarizer

sample_rr_output = json.load(open('samples/sample_rhetorical_role_output.json'))

summarizer = ExtractiveSummarizer(use_gpu=True, verbose=False)
summaries = summarizer(sample_rr_output)
```

Result:

```
[{'facts': 'xxxx',
  'arguments': 'xxxx',
  'ANALYSIS': 'xxxx',
  'issue': 'xxxx',
  'decision': 'xxxx',
  'PREAMBLE': 'xxxx'}]
 ```

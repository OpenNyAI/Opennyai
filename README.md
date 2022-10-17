<a href="https://github.com/OpenNyAI/Opennyai"><img src="https://github.com/OpenNyAI/Opennyai/raw/master/asset/final-logo-01.jpeg" width="190" height="65" align="right" /></a>

# Opennyai : An efficient NLP Pipeline for Indian Legal documents

[![PyPI version](https://badge.fury.io/py/opennyai.svg)](https://pypi.org/project/opennyai/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/OpenNyAI/Opennyai)

Opennyai is a natural language preprocessing framework made with SpaCy. Its pipeline has achieved State-of-the-Art
performance on Named entity recognition on Indian legal NER.
Try [demo](https://huggingface.co/opennyaiorg/en_legal_ner_trf)

This library provides unified access to three AI models developed by us which focus on Indian court judgments:

* [Named Entity Recognition (NER)](https://github.com/Legal-NLP-EkStep/legal_NER)
* [Rhetorical Role prediction](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline)
* [Extractive Summarizer](https://github.com/Legal-NLP-EkStep/judgment_extractive_summarizer)

# 🔧 1. Installation

To get started using opennyai first create a new python virtual environment:

```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
Install it using pip by running the following line in your terminal

```bash
pip install -U opennyai
```

Note: if you want to utilize spacy with GPU please install [Cupy](https://anaconda.org/conda-forge/cupy)
/[cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) dependency of appropriate version. For spacy installation with
cupy refer to [page](https://spacy.io/usage)

Remember you need spacy of 3.2.4 version for models to work perfectly.

# 📖 2. [Documentation](https://opennyai.readthedocs.io/en/latest/)

# 👩‍💻 3. Usage
To run the 3 OpenNyAI models on judgment texts of your choice please run following python code
```python
from opennyai import Pipeline
from opennyai.utils import Data,get_text_from_indiankanoon_url

###### Get court judgment texts on which to run the AI models
text1 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/811682/')
text2 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/1386912/')
texts_to_process = [text1,text2] ### you can also load your text files directly into this
data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

use_gpu = True #### If you have access to GPU then set this to True else False
###### Choose which of the AI models you want to run from the 3 models 'NER', 'Rhetorical_Role','Summarizer'
pipeline = Pipeline(components = ['NER', 'Rhetorical_Role','Summarizer'],use_gpu=use_gpu) #E.g. If just Named Entity is of interest then just select 'NER'
results = pipeline(data)
```
The output of each model is present in following keys of each element of the output
```python
results[0]['ner_annotations'][0]['result'] ## shows the NER model output for the first judgment text
results[0]['rr_annotations'][0]['result']  ## shows the Rhetorical Roles model output for the first judgment text
results[0]['summary'] ## shows Summary for each of the Rheorical Role for first judgment text 
```


### 3.1 Running each of the 3 AI models individually
If you need to more customizations on the output of each of the models then you can also run each of the models individually
####  3.1.1 Run NER model only
For detailed information about the NER model, please refer to [legal_NER git repo](https://github.com/Legal-NLP-EkStep/legal_NER).
Use following python to extract entities from single court judgment.

```python
import opennyai.ner as InLegalNER
from opennyai.utils import Data,get_text_from_indiankanoon_url

text = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/811682/')
data = Data(text) #### Data object for preprocessing
NER_model = InLegalNER.load('en_legal_ner_trf', use_gpu=True)  ## load spacy pipeline for Named Entity Recognition
ner_output = NER_model(data, do_sentence_level=True,do_postprocess=True)   
identified_entites = [(ent, ent.label_) for ent in ner_output.ents]
```

Result:

```
[(Section 319, 'PROVISION'),
 (Cr.P.C., 'STATUTE'),
 (G. Sambiah, 'RESPONDENT'),
 (20th June 1984, 'DATE')]
 ```

To visualize the NER results please run 
```python
from spacy import displacy
from opennyai.ner.ner_utils import ner_displacy_option
displacy.serve(ner_output, style='ent',port=8080,options=ner_displacy_option)
```
Please click on the link displayed in the console to see the annotated entities

To get result in json format with span information:

```python
json_result = InLegalNER.get_json_from_spacy_doc(ner_output)
```

Note: You can import generated json to label studio and visualize all the details about the postprocessing



#### 3.1.2 Run Rhetorical Role model only
For detailed information about the Rhetorical Roles model, please refer to [git repo](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline).

Use following python to get structure of 2 court judgments using sentence rhetorical roles.
```python
from opennyai import RhetoricalRolePredictor
from opennyai.utils import Data,get_text_from_indiankanoon_url

text1 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/811682/')
text2 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/1386912/')
texts_to_process = [text1,text2] ### you can also load your text files directly into this
data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

rr_model = RhetoricalRolePredictor(use_gpu=True)
rr_output = rr_model(data)
```

#### 3.1.3 Run Summarizer model only
For detailed information about extactive summarization of court judgment , please refer to [git repo](https://github.com/Legal-NLP-EkStep/judgment_extractive_summarizer).


Summarizer model needs Rhetorical Role model output as input. Hence Rhetorical Role prediction model needs to run before Summarizer model rune.

To use Summarizer model simply execute:

```python
from opennyai import RhetoricalRolePredictor,ExtractiveSummarizer
from opennyai.utils import Data,get_text_from_indiankanoon_url

text1 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/811682/')
text2 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/1386912/')
texts_to_process = [text1,text2] ### you can also load your text files directly into this
data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

rr_model = RhetoricalRolePredictor(use_gpu=True)
rr_output = rr_model(data)


summarizer = ExtractiveSummarizer(use_gpu=True, verbose=False)
summaries = summarizer(rr_output)
```

Result:

```
{'id': 'ExtractiveSummarizer_xxxxxxx]',
  'summaries': {'facts': 'xxxx',
  'arguments': 'xxxx',
  'ANALYSIS': 'xxxx',
  'issue': 'xxxx',
  'decision': 'xxxx',
  'PREAMBLE': 'xxxx'}]
 ```

## 3.2 Advanced Usage
### 3.2.1 Trade off between run time and accuracy for data preprocessing 
Data Preprocessing performs tasks like sentence splitting , splitting the preamble and judgment. Performance of this preprocessing critically determines the performance of AI models.
We recommend using 'en_core_web_trf' for preprocessing of the data, but it can be slow.
Available preprocessing models are 'en_core_web_trf' (slowest but best accuracy), 'en_core_web_md', 'en_core_web_sm'(fastest but less accurate)

### 3.2.2 Trade off between run time and accuracy for NER 
You can choose from following NER models in InLegalNER.load() depending on your accuracy and run time needs
* en_legal_ner_trf (This model provides the highest accuracy)
* en_legal_ner_sm (This model provides the highest efficiency)

If do_sentence_level=True (recommended) then single sentence is passed through the model which gives better results. If set to False then multiple sentences which fit the max length of 512 tokens are passed through the model. This reduces run time but gives poor accuracy.

<a href="https://github.com/OpenNyAI/Opennyai"><img src="https://github.com/OpenNyAI/Opennyai/raw/master/asset/final-logo-01.jpeg" width="190" height="65" align="right" /></a>

# Opennyai : An efficient NLP Pipeline for Indian Legal documents

[![Current Release Version](https://img.shields.io/github/release/OpenNyAI/opennyai.svg?style=flat-square&logo=github)](https://github.com/OpenNyAI/Opennyai/releases)
[![PyPI version](https://img.shields.io/pypi/v/opennyai.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/opennyai/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/OpenNyAI/Opennyai)
[![python version](https://img.shields.io/badge/Python-<3.11-blue)](https://github.com/OpenNyAI/Opennyai)
[![Downloads](https://pepy.tech/badge/opennyai)](https://github.com/OpenNyAI/Opennyai)

Opennyai is a python library for natural language preprocessing on Indian legal texts.

This library provides unified access to the following 3 pre-trained AI models developed by OpenNyAI which focus on
Indian court
judgments:

* Named Entity Recognition (NER): [GitHub](https://github.com/Legal-NLP-EkStep/legal_NER)
  , [paper](https://arxiv.org/pdf/2211.03442.pdf)
* Judgment Structuring using Sentence Rhetorical
  Roles: [GitHub](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline)
  , [paper](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.470.pdf)
* Extractive Summarizer: [GitHub](https://github.com/Legal-NLP-EkStep/judgment_extractive_summarizer)

This library is mainly for running the pretrained models on your custom input judgments text. For more details about
data and model training, please refer to individual git repo links.

# 🔧 1. Installation

To get started using opennyai first create a new python virtual environment using [conda](https://www.anaconda.com/):

Supports python 3.8, 3.9, 3.10

```bash
conda create -n opennyai python=3.8
conda activate opennyai
```

Install it using pip by running the following line in your terminal

```bash
pip install -U opennyai
```

#### For GPU support

If you want to utilize spacy with GPU please install [Cupy](https://anaconda.org/conda-forge/cupy) and
[cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) dependency of appropriate version.

```bash
conda install cudatoolkit==<your_cuda_version> #### E.g. cudatoolkit==11.2
pip install cupy-cuda<your_cuda_version> ##### E.g. cupy-cuda112
```

In case of any issue with installation please refer to [spacy installation with cupy](https://spacy.io/usage)

Remember you need spacy of 3.2.4 version for models to work perfectly.

# 📖 2. Documentation

Please refer to the [Documentation](https://opennyai.readthedocs.io/en/latest/index.html#) for more details.

# 👩‍💻 3. Usage

To run the 3 OpenNyAI models on judgment texts of your choice please run following python code

```python
from opennyai import Pipeline
from opennyai.utils import Data
import urllib

# Get court judgment texts on which to run the AI models
text1 = urllib.request.urlopen(
    'https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt').read().decode()
text2 = urllib.request.urlopen(
    'https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment2.txt').read().decode()

# you can also load your text files directly into this
texts_to_process = [text1, text2]

# create Data object for data  preprocessing before running ML models
data = Data(texts_to_process)

# If you have access to GPU then set this to True else False
use_gpu = True

# Choose which of the AI models you want to run from the 3 models 'NER', 'Rhetorical_Role','Summarizer'. E.g. If just Named Entity is of interest then just select 'NER'

pipeline = Pipeline(components=['NER', 'Rhetorical_Role', 'Summarizer'], use_gpu=use_gpu, verbose=True)

results = pipeline(data)
```

The output of each model is present in following keys of each element of the output

```python
results[0][
    'annotations']  ## shows the result of model at sentence level, each entry will have entities, rhetorical role, and other details
results[0]['summary']  ## shows Summary for each of the Rheorical Role for first judgment text 
```

For more details on usage please refer to the [documentation](https://opennyai.readthedocs.io/en/latest/index.html#)

Google Colab Notebook
----------------------
We encourage you to use the local machine installation of opennyai library instead of Google colab as it works with python 3.8. Since Google colab works with higher versions of python, opennyai installation does not work.

| Description               | Link  |
|---------------------------|-------|
| Run Inference          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rNA6XVyD-GCTd0YtosjiKON_p9bGuVwz) |

Visualization of outputs
-----------------------
We encourage users to use [our webapp](https://summarizer-fer6v2lowq-uc.a.run.app/) for visualizing the results for a
judgment of your choice.

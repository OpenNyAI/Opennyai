Installation
============

To get started using opennyai first create a new conda environment:

.. code-block::

    conda create -n opennyai python=3.8
    conda activate opennyai

Install it using pip by running the following line in your terminal

.. code-block::

    pip install opennyai

For GPU support
---------------
If you want to utilize spacy with GPU please install `Cupy <https://anaconda.org/conda-forge/cupy>`_ and
`cudatoolkit <https://anaconda.org/anaconda/cudatoolkit>`_ dependency of appropriate version.

.. code-block::

    conda install cudatoolkit==<your_cuda_version> #### E.g. cudatoolkit==11.2
    pip install cupy-cuda<your_cuda_version> ##### E.g. cupy-cuda112


In case of any issue with installation please refer to `spacy installation with cupy <https://spacy.io/usage>`_

Remember you need spacy of 3.2.4 version for models to work perfectly.




Run All 3 AI models on Input Judgment Texts
==============================
To run the 3 OpenNyAI models on judgment texts of your choice please run following python code

.. code-block:: python

    from opennyai import Pipeline
    from opennyai.utils import Data,get_text_from_indiankanoon_url

    ###### Get court judgment texts on which to run the AI models
    text1 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/542273/')
    text2 = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/82089984/')
    texts_to_process = [text1,text2] ### you can also load your text files directly into this
    data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

    use_gpu = True #### If you have access to GPU then set this to True else False
    ###### Choose which of the AI models you want to run from the 3 models 'NER', 'Rhetorical_Role','Summarizer'
    pipeline = Pipeline(components = ['NER', 'Rhetorical_Role','Summarizer'],use_gpu=use_gpu) #E.g. If just Named Entity is of interest then just select 'NER'
    results = pipeline(data)


The predictions of each of the models is added at the sentence level.

For each of the sentence in an output,

* 'labels' provide predicted rhetorical role.

* 'in_summary' denoted if this sentence is selected in the summary.

* 'entities' provide the list of extracted named entities in that sentence

.. code-block:: python

    results[0]['annotations']

The AI generated summary by rhetorical roles can be accessed via

.. code-block:: python

    results[0]['summary']

The extracted named entities can be visualized using

.. code-block:: python

    from spacy import displacy
    from opennyai.ner.ner_utils import ner_displacy_option
    displacy.serve(pipeline._ner_model_output[0], style='ent', port=8080, options=ner_displacy_option)


Try on Google Colab
==============================
`Open In Colab <https://colab.research.google.com/drive/1rNA6XVyD-GCTd0YtosjiKON_p9bGuVwz>`_

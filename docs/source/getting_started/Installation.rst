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
    from opennyai.utils import Data
    import urllib

    ###### Get court judgment texts on which to run the AI models
    text1 = urllib.request.urlopen('https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt').read().decode()
    text2 = urllib.request.urlopen('https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment2.txt').read().decode()
    texts_to_process = [text1,text2] ### you can also load your text files directly into this
    data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

    use_gpu = True #### If you have access to GPU then set this to True else False
    ###### Choose which of the AI models you want to run from the 3 models 'NER', 'Rhetorical_Role','Summarizer'
    pipeline = Pipeline(components = ['NER', 'Rhetorical_Role','Summarizer'],use_gpu=use_gpu) #E.g. If just Named Entity is of interest then just select 'NER'
    results = pipeline(data)


Extra parameters for Pipeline:

* **components (list):** Models that you want to run over your input judgements
* **use_gpu (bool):** Functionality to give a choice whether to use GPU for inference or not. Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in documentation
* **verbose (bool):** Set it to True if you want to see progress bar/updates while processing happens
* **ner_model_name (string):** Accepts a model name of spacy as InLegalNER that will be used for NER inference available models are 'en_legal_ner_trf', 'en_legal_ner_sm'. 'en_legal_ner_trf' has best accuracy but can be slow, on the other hand 'en_legal_ner_sm' is fast but less accurate.
* **ner_mini_batch_size (int):** This accepts an int as batch size for processing of a document, if length of document is bigger that given batch size it will be chunked and then processed.
* **ner_do_sentence_level (bool):** To perform inference at sentence level or not, at sentence level it better accuracy. We recommend setting this to True.
* **ner_do_postprocess (bool):** To perform post-processing over processed doc. We recommend to set this to True.
* **ner_statute_shortforms_path (path):** It is the path of the csv file if the user wants to provide predefined shortforms to create statute clusters.The csv should have 2 columns namely 'fullforms' and 'shortforms' where 'fullforms' contain the full name of the statute eg. 'code of criminal procedure' and shortforms contain the acronym that can be present in the judgment eg.'crpc'.Each row represents a fullform,shortform pair.
* **summarizer_summary_length (float):** Give you the functionality to choose the length of generated summary. Default is 0 which will set it to adaptive length selection. Valid input lie in range(0-1)



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

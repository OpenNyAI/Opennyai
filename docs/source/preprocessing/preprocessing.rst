Preprocessing Judgment Text
===========================
Judgment texts need to be preprocessed before running the AI models

Preprocessing Activities:
------------------------
Following preprocessing activities are performed using spacy pretrained model

1. Separating preamble from judgment text

2. Sentence splitting of judgment text

3. Convert upper case words in preamble to title case

4. Replace newline characters within a sentence with space in judgment text

The preprocessing is done using Data object.

.. code-block:: python

    texts_to_process = [text1,text2]
    data = Data(texts_to_process,preprocessing_nlp_model='en_core_web_trf')


The preprocessing is lazy evaluated.

Trade-Off between Preprocessing Accuracy and Run Time
-----------
One can choose which spacy pretrained model to use for preprocessing while creating Data object using parameter `preprocessing_nlp_model`.
The choice of preprocessing model critically determines the performance of AI models. We recommend using 'en_core_web_trf' for preprocessing of the data, but it can be slow. Available preprocessing models are 'en_core_web_trf' (slowest but best accuracy), 'en_core_web_md', 'en_core_web_sm'(fastest but less accurate)

Additional Parameters while creating Data object
-----------
* mini_batch_size (int): This accepts an int as batch size for processing of a document, if length of document is bigger that given batch size it will be chunked and then processed.

* use_gpu (bool): Functionality to give a choice whether to use GPU for processing or not Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in documentation

* use_cache (bool): Set it to true if you want to enable caching while preprocessing. Always set this to True.

* verbose (bool): Set it to if you want to see progress bar while processing happens

* file_ids (list): List of custom file ids to use for documents
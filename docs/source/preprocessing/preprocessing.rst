Preprocessing Judgment Text
===========================
Before passing input judgment texts through the AI models, one needs to run preprocessing on them.

Preprocessing Activities:
------------------------
Following preprocessing activities are performed using spacy pretrained model
1. Separating preamble from judgment text
2. Sentence splitting

The preprocessing is done using Data object.
.. code-block:: python

    texts_to_process = [text1,text2]
    data = Data(texts_to_process,preprocessing_nlp_model='en_core_web_trf')


The preprocessing is lazy evaluated. This means that when some action is taken on the Data object then the actual processing happens.

Trade-Off between Preprocessing Accuracy and Run Time
-----------
One can choose which spacy pretrained model to use for preprocessing while creating Data object using parameter `preprocessing_nlp_model`.
The choice of preprocessing model critically determines the performance of AI models. We recommend using 'en_core_web_trf' for preprocessing of the data, but it can be slow. Available preprocessing models are 'en_core_web_trf' (slowest but best accuracy), 'en_core_web_md', 'en_core_web_sm'(fastest but less accurate)
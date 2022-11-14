Welcome to OpenNyAI's documentation!
===================================

**OpenNyAI** library is a framework for natural language processing on Indian legal texts. This python library provides unified access to the inference of following 3 AI models developed by `OpenNyAI <https://opennyai.org/>`_ which focus on Indian court judgments.

* `Named Entity Recognition (NER) <https://github.com/Legal-NLP-EkStep/legal_NER>`_
* `Judgment structuring using sentence Rhetorical Role prediction <https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline>`_
* `Extractive Summarizer <https://github.com/Legal-NLP-EkStep/judgment_extractive_summarizer>`_

This library is mainly for running the pretrained models on your custom input judgments text. For more details about data and model training, please refer to individual git repo links.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   getting_started/about_opennyai
   getting_started/Installation
   getting_started/hosted_webapp_api

.. toctree::
   :maxdepth: 1
   :caption: Preprocessing:

   preprocessing/preprocessing

.. toctree::
   :maxdepth: 1
   :caption: Named Entity Recognition:

   ner/legal_named_entities

.. toctree::
   :maxdepth: 1
   :caption: Judgment Structuring:

   rr/rr_structure

.. toctree::
   :maxdepth: 1
   :caption: Judgment Summarization:

   summariser/summariser
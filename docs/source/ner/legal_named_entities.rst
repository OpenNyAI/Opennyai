Which Legal Named Entities are extracted?
==============================
Named Entities Recognition is commonly studied problem in Natural Language Processing and many pre-trained models are publicly available. However legal documents have peculiar named entities like names of petitioner, respondent, court, statute, provision, precedents, etc. These entity types are not recognized by standard Named Entity Recognizer like spacy. Hence there is a need to develop a Legal NER model. Due to peculiarity of Indian legal processes and terminologies used, it is important to develop separate legal NER for Indian court judgment texts.

Some entities are extracted from Preamble of the judgements and some from judgement text. Preamble of judgment contains formatted metadata like names of parties, judges, lawyers,date, court etc. The text following preamble till the end of the judgment is called as the "judgment". Below is an example

.. image:: NER_example.png

Following legal entities are extracted from input court judgment.

 =============== ===================== ====================================================================================================================================================================== ===
  Named Entity    Extract From          Description
 =============== ===================== ====================================================================================================================================================================== ===
  COURT           Preamble, Judgment    Name of the court which has delivered the current judgement if extracted from Preamble. Name of any court mentioned if extracted from judgment sentences.
  PETITIONER      Preamble, Judgment    Name of the petitioners / appellants /revisionist  from current case
  RESPONDENT      Preamble, Judgment    Name of the respondents / defendents /opposition from current case
  JUDGE           Premable, Judgment    Name of the judges from current case  if extracted from preamble. Name of the judges of the current as well as previous cases if extracted from judgment sentences.
  LAWYER          Preamble              Name of the lawyers from both the parties
  DATE            Judgment              Any date mentioned in the judgment
  ORG             Judgment              Name of organizations mentioned in text apart from court. E.g. Banks, PSU, private companies, police stations, state govt etc.
  GPE             Judgment              Geopolitical locations which include names of countries,states,cities, districts and villages
  STATUTE         Judgment              Name of the act or law mentioned in the judgement
  PROVISION       Judgment              Sections, sub-sections, articles, orders, rules under a statute
  PRECEDENT       Judgment              All the past court cases referred in the judgement as precedent. Precedent consists of party names + citation(optional) or case number (optional)
  CASE\_NUMBER    Judgment              All the other case numbers mentioned in the judgment (apart from precedent) where party names and citation is not provided
  WITNESS         Judgment              Name of witnesses in current judgment
  OTHER_PERSON    Judgment              Name of the all the person that are not included in petitioner,respondent,judge and witness
 =============== ===================== ====================================================================================================================================================================== ===


More detailed definitions with examples can be found `here <https://docs.google.com/presentation/d/e/2PACX-1vSpWE_Qk9X_wBh7xJWPyYcWcME3ZBh_HmqeZOx58oMLyJSi0Tn0-JMWKI-HsQIRuUTbQHPql6MlU7OS/pub?start=false&loop=false&delayms=3000>`_
For more details about training data and code used for training , please refer to `legal_NER git repo <https://github.com/Legal-NLP-EkStep/legal_NER>`_.

Extract Named Entities from Judgment Text
======================
Use following python to extract entities from single court judgment. For running all 3 AI models together on input text, please refer to :ref:`here<Run All 3 AI models on Input Judgment Texts>`.

.. code-block:: python

    import opennyai.ner as InLegalNER
    from opennyai.utils import Data,get_text_from_indiankanoon_url

    text = get_text_from_indiankanoon_url('https://indiankanoon.org/doc/811682/')
    data = Data(text) #### Data object for preprocessing
    NER_model = InLegalNER.load(model_name='en_legal_ner_trf', use_gpu=True)  ## load spacy pipeline for Named Entity Recognition
    ner_output = NER_model(data, do_sentence_level=True,do_postprocess=True)
    identified_entites = [(ent, ent.label_) for ent in ner_output.ents]

Output of NER model is a spacy doc and identified_entities is list of entities extracted.

.. code-block:: python

    [(Section 319, 'PROVISION'),
     (Cr.P.C., 'STATUTE'),
     (G. Sambiah, 'RESPONDENT'),
     (20th June 1984, 'DATE')]

Important parameters while loading NER model
--------------------
model_name (string): Accepts a model name of spacy as InLegalNER that will be used for NER inference available models are 'en_legal_ner_trf', 'en_legal_ner_sm'. 'en_legal_ner_trf' has best accuracy but can be slow, on the other hand 'en_legal_ner_sm' is fast but less accurate.

use_gpu (bool): Functionality to give a choice whether to use GPU for inference or not. Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in documentation

Important parameters while inferring NER model
--------------------
do_sentence_level (bool): To perform inference at sentence level or not, at sentence level it better accuracy. We recommend setting this to True.

do_postprocess (bool): To perform post-processing over processed doc. We recommend to set this to True.

mini_batch_size (int): This accepts an int as batch size for processing of a document, if length of document is bigger that given batch size it will be chunked and then processed.

verbose (bool): Set it to if you want to see progress bar while processing happens

Post Processing of extracted Named Entities
======================
Since the document level context was not used duiring annotation,it is important to capture the document level context while inference. This can be done via postprocessing using rules.

To perform postprocessing on the extracted entities specify `do_postprocessing=True`.

The postprocessing is done on these entities:

1. `Precedents`: Same precedent can be written in multiple forms in a judgment. E.g. with citation,without
citation,only petitioner's name supra etc. After postprocessing,all the precedents referring to the same case
are  clustered together and an extension 'precedent_clusters' is
added to the doc which is a dict where the keys are the head of the cluster (longest precedent) and value
is a list of all the precedents in that cluster. To access the list, use

`doc._.precedent_clusters`

For example
 [{Madhu Limaye v. State of Mahrashtra: [Madhu Limaye v. State of Mahrashtra, Madhu Limaye v. State of Maharashtra, Madhu Limaye, Madhu Limaye, Madhu Limaye]}]

2. `Statute`: The same statute can be written in multiple ways in a judgment. E.g. 'Indian Penal Code', 'IPC', 'Indian Penal Code, 1860' etc.
We cluster all such statutes and assign a head of such cluster as the full form of the statute. The statute clusters can be accessed by
`doc._.provision_statute_clusters`

3. `Provision-Statute`: Every provision should have an associated statute but sometimes the
corresponding statutes are not mentioned explicitly in the judgment. Postprocessing contains a
set of rules to identify statute for each provision and the extension 'provision_statute_clusters'
is added to the doc which is a list of tuples, each tuple contains provision-statute-normalised provision text eg. (362,IPC,'Section 362') .It can be
used by:

`doc._.provision_statute_clusters`

For example
[(Section 369, Crpc, 'Section 369'), (Section 424, Crpc, 'Section 424')]

4. `Other person/Org` : Same entities can be tagged with different classes in different sentences of
the same judgment due to sentence level context. E.g. 'Amit Kumar' can be  a petitioner
in the preamble but later in the judgment is marked as 'other_person'. So,we reconcile these entities
based on their relative importance i.e. 'Amit Kumar' will be marked as petitioner in the
whole judgment.



Visualization of extracted Named Entities
======================
To visualize the NER result on single judgment text please run

.. code-block:: python

    from spacy import displacy
    from opennyai.ner.ner_utils import ner_displacy_option
    displacy.serve(ner_output, style='ent',port=8080,options=ner_displacy_option)


Please click on the link displayed in the console to see the annotated entities.

Storing extracted Named Entities to a file
======================
1. To get result in json format with span information:


.. code-block:: python

    json_result = InLegalNER.get_json_from_spacy_doc(ner_output)


Note: You can import generated json to label studio and visualize all the details about the postprocessing

2. To get result in csv file with linked entities (coming soon)



Huggingface Models
======================
These models are also published on huggingface

`en_legal_ner_trf <https://huggingface.co/opennyaiorg/en_legal_ner_trf>`_ and `en_legal_ner_sm <https://huggingface.co/opennyaiorg/en_legal_ner_sm>`_
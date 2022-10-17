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

Extract Named Entities from Judgment Text
======================


Visualization of extracted Named Entities
======================


Post Processing of extracted Named Entities
======================

Huggingface Models
======================

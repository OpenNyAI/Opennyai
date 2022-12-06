Automatic Summarization of Court Judgments
====================================
Court judgments can be very long and it is a common practice for legal publishers to create headnotes of judgments. E.g. `sample headnote <https://main.sci.gov.in/judgment/judis/5268.pdf>`_.
The process of creating headnotes is  manual and based on the certain rules and patterns. With advances in Artifical Intelligence, we can create automatically summaries of long text and then an expert to correct it to create final summary. This will drastically reduce the time needed for creation of headnotes and make the process more consistent. AI model can also learn from the feedback given by the expert and keep on improving the results.

Structure of Judgment Summary
====================================
While standard way of writing headnotes captures the important aspects of the judgement like HELD, experts believe that it is not the best style of writing summaries. E.g. it is difficult to establish if the facts of a new case are similar to facts of an old case by reading headnotes of the old case.

So we have come up with revised structure of writing summaries. Summary will have 5 sections Facts summary, Arguments summary, Issue summary, Analysis Summary and Decision Summary. Leveraging our previous work on `structuring court judgements <https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline>`_, we can automatically predict Rhetorical Roles for each sentence and then create this sectionwise summary. The following table shows which rhetorical roles to expect in each of the summary sections.

 ================== ===================================================================================
  Summary Section    Rhetorical Roles
 ================== ===================================================================================
  Facts              Facts, Ruling by Lower Court
  Issue              Issues
  Arguments          Argument by Petitioner, Argument by Respondent
  Analysis           Analysis, Statute, Precedent Relied, Precedent Not Relied, Ratio of the decision
  Decision           Ruling by Present Court
 ================== ===================================================================================

We believe this structure of writing summaries is better suited for Legal Research and Infomation Extraction. This will also improve the readability of the summaries.

Extractive summarization using Rhetorical Roles
------------------
There are two styles of creating summaries viz. Extractive & Abstractive. Extractive summaries pick up important sentences as-is and put them in order for creating final summary. Abstractive summarization on the other hand paraphrases the important information to create crisp summary in its own words. While abstractive summaries are more useful, they are harder to create and evaluate. Hence, as first step we focus on extractive summarization which will pick up most important sentences and arrange them in the structure described above. Once this task is done correctly, then we can focus on the abstractive summarization

Which rhetorical roles are summarized?
------------------
We empirically found out that "Issues" and "Decision" written in original judgement are very crisp and rich in information. So we do not try to summarize them. We carry forward all the sentences with these 2 roles directly into the summary. "Preamble" is important in setting the context of case and also copied to summary.  For remaining rhetorical roles, we rank the sentences in descending order of importance as predicted by the AI model and choose the top ones as described in section 5.

Create summary of input judgment text
======================
Summarizer model needs Rhetorical Role model output as input. Hence Rhetorical Role prediction model needs to run before Summarizer model rune.

To use Summarizer model simply execute code below. For running all 3 AI models together on input text, please refer :ref:`here<Run All 3 AI models on Input Judgment Texts>` .

.. code-block:: python

    from opennyai import Pipeline
    from opennyai.utils import Data
    import urllib

    ###### Get court judgment texts on which to run the AI models
    text1 = urllib.request.urlopen('https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt').read().decode()
    text2 = urllib.request.urlopen('https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment2.txt').read().decode()
    texts_to_process = [text1,text2] ### you can also load your text files directly into this
    data = Data(texts_to_process)  #### create Data object for data  preprocessing before running ML models

    pipeline = Pipeline(components=['Rhetorical_Role', 'Summarizer'], use_gpu=use_gpu, verbose=True, summarizer_summary_length=0.0)

    results = pipeline(data)

    json_result_doc_1 = results[0]
    summaries_doc_1 = results[0]['summary']


Result:

.. code-block:: python

    {'id': 'ExtractiveSummarizer_xxxxxxx]',
      'summaries': {'facts': 'xxxx',
      'arguments': 'xxxx',
      'ANALYSIS': 'xxxx',
      'issue': 'xxxx',
      'decision': 'xxxx',
      'PREAMBLE': 'xxxx'}]


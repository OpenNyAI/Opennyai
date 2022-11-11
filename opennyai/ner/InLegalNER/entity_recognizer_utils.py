import warnings

import spacy

from opennyai.utils.sentencizer import process_nlp_in_chunks


def extract_entities_from_judgment_text(to_process, legal_nlp, mini_batch_size, do_sentence_level=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        judgement = to_process['judgement_doc']
        preamble = to_process['preamble_doc']
        if do_sentence_level:
            doc_judgement = get_sentence_docs(judgement, legal_nlp)
            doc_preamble = process_nlp_in_chunks(preamble.text, mini_batch_size, legal_nlp)
        else:
            doc_judgement = process_nlp_in_chunks(judgement.text, mini_batch_size, legal_nlp)
            doc_preamble = process_nlp_in_chunks(preamble.text, mini_batch_size, legal_nlp)

        ######### Combine preamble doc & judgement doc
        combined_doc = spacy.tokens.Doc.from_docs([doc_preamble, doc_judgement])

    return combined_doc


def get_sentence_docs(doc_judgment, nlp_judgment):
    sentences = [sent.text for sent in doc_judgment.sents]
    docs = []
    for doc in nlp_judgment.pipe(sentences):
        docs.append(doc)
    combined_docs = spacy.tokens.Doc.from_docs(docs)
    return combined_docs

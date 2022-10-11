import spacy, warnings
from opennyai.utils.sentencizer import split_main_judgement_to_preamble_and_judgement


def extract_entities_from_judgment_text(txt, legal_nlp, preamble_splitting_nlp, do_sentence_level=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preamble, judgement = split_main_judgement_to_preamble_and_judgement(text=txt,
                                                                             sentence_splitting_nlp=preamble_splitting_nlp,
                                                                             return_nlp_doc=do_sentence_level)
        if do_sentence_level:
            doc_judgement = get_sentence_docs(judgement, legal_nlp)
            doc_preamble = legal_nlp(preamble.text)
        else:
            doc_judgement = legal_nlp(judgement)
            doc_preamble = legal_nlp(preamble)

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

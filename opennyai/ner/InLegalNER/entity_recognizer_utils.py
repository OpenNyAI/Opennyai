import re
import copy
import spacy, warnings


def remove_unwanted_text(text):
    '''Looks for pattern  which typically starts the main text of jugement.
    The text before this pattern contains metadata like name of paries, judges and hence removed'''
    pos_list = []
    len = 0
    pos = 0
    pos_list.append(text.find("JUDGMENT & ORDER"))
    pos_list.append(text.find("J U D G M E N T"))
    pos_list.append(text.find("JUDGMENT"))
    pos_list.append(text.find("O R D E R"))
    pos_list.append(text.find("ORDER"))

    for i, p in enumerate(pos_list):

        if p != -1:
            if i == 0:
                len = 16
            elif i == 1:
                len = 15
            elif i == 2:
                len = 8
            elif i == 3:
                len = 9
            elif i == 4:
                len = 5
            pos = p + len
            break

    return pos


def get_keyword_based_preamble_end_char_offset(text):
    preamble_end_keywords = ["JUDGMENT", "ORDER", "J U D G M E N T", "O R D E R", "JUDGMENT & ORDER", "COMMON ORDER",
                             "ORAL JUDGMENT"]
    preamble_end_char_offset = 0

    ### search for preamble end keywords on new lines
    for preamble_keyword in preamble_end_keywords:
        match = re.search(r'\n\s*' + preamble_keyword + r'\s*\n', text)
        if match:
            preamble_end_char_offset = match.span()[1]
            break

    #### if not found then search for the keywords anywhere
    if preamble_end_char_offset == 0:
        for preamble_keyword in preamble_end_keywords:
            match = re.search(preamble_keyword, text)
            if match:
                preamble_end_char_offset = match.span()[1]
                break
    return preamble_end_char_offset


def convert_upper_case_to_title(txt):
    ########### convert the uppercase words to title case for catching names in NER
    title_tokens = []
    for token in txt.split(' '):
        title_subtokens = []
        for subtoken in token.split('\n'):
            if subtoken.isupper():
                title_subtokens.append(subtoken.title())
            else:
                title_subtokens.append(subtoken)
        title_tokens.append('\n'.join(title_subtokens))
    title_txt = ' '.join(title_tokens)
    return title_txt


def guess_preamble_end(truncated_txt, nlp):
    ######### Guess the end of preamble using hueristics
    preamble_end = 0
    max_length = 20000
    tokens = nlp.tokenizer(truncated_txt)
    if len(tokens) > max_length:
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        nlp_docs = [nlp(i.text) for i in chunks]
        truncated_doc = spacy.tokens.Doc.from_docs(nlp_docs)
    else:
        truncated_doc = nlp(truncated_txt)
    successive_preamble_pattern_breaks = 0
    preamble_patterns_breaks_theshold = 1  ####### end will be marked after these many consecutive sentences which dont match preamble pattern
    sent_list = [sent for sent in truncated_doc.sents]
    for sent_id, sent in enumerate(sent_list):
        ###### check if verb is present in the sentence
        verb_exclusions = ['reserved', 'pronounced', 'dated', 'signed']
        sent_pos_tag = [token.pos_ for token in sent if token.lower_ not in verb_exclusions]
        verb_present = 'VERB' in sent_pos_tag

        ###### check if uppercase or title case
        allowed_lowercase = ['for', 'at', 'on', 'the', 'in', 'of']
        upppercase_or_titlecase = all(
            [token.text in allowed_lowercase or token.is_upper or token.is_title or token.is_punct for token in sent if
             token.is_alpha])

        if verb_present and not upppercase_or_titlecase:
            successive_preamble_pattern_breaks += 1
            if successive_preamble_pattern_breaks > preamble_patterns_breaks_theshold:
                preamble_end = sent_list[sent_id - preamble_patterns_breaks_theshold - 1].end_char
                break
        else:
            if successive_preamble_pattern_breaks > 0 and (verb_present or not upppercase_or_titlecase):
                preamble_end = sent_list[sent_id - preamble_patterns_breaks_theshold - 1].end_char
                break
            else:
                successive_preamble_pattern_breaks = 0
    return preamble_end


def seperate_and_clean_preamble(txt, preamble_splitting_nlp):
    ########## seperate preamble from judgment text

    ######## get preamble end offset based on keywords
    keyword_preamble_end_offset = get_keyword_based_preamble_end_char_offset(txt)
    if keyword_preamble_end_offset == 0:
        preamble_end_offset = 5000  ######## if keywords not found then set arbitrarty value to reduce time for searching
    else:
        preamble_end_offset = keyword_preamble_end_offset + 200  ######## take few more characters as judge names are written after JUDGEMENT keywords
    truncated_txt = txt[:preamble_end_offset]
    guessed_preamble_end = guess_preamble_end(truncated_txt, preamble_splitting_nlp)

    if guessed_preamble_end == 0:
        preamble_end = keyword_preamble_end_offset
    else:
        preamble_end = guessed_preamble_end

    preamble_txt = txt[:preamble_end]
    title_txt = convert_upper_case_to_title(preamble_txt)
    return title_txt, preamble_end


def extract_entities_from_judgment_text(txt, legal_nlp, preamble_splitting_nlp, do_sentence_level=True):
    ######### Seperate Preamble and judgment text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preamble_text, preamble_end = seperate_and_clean_preamble(txt, preamble_splitting_nlp)

        ########## process main judgement text
        judgement_text = txt[preamble_end:]
        #####  replace new lines in middle of sentence with spaces.
        judgement_text = re.sub(r'(\w[ -]*)(\n+)', r'\1 ', judgement_text)
        if do_sentence_level:
            judgment_doc = preamble_splitting_nlp(judgement_text)
            doc_judgment = get_sentence_docs(judgment_doc, legal_nlp)
        else:
            doc_judgment = legal_nlp(judgement_text)
        ######### process preamable
        doc_preamble = legal_nlp(preamble_text)

        ######### Combine preamble doc & judgement doc
        combined_doc = spacy.tokens.Doc.from_docs([doc_preamble, doc_judgment])

    return combined_doc


def get_sentence_docs(doc_judgment, nlp_judgment):
    sentences = [sent.text for sent in doc_judgment.sents]
    docs = []
    for doc in nlp_judgment.pipe(sentences):
        docs.append(doc)
    combined_docs = spacy.tokens.Doc.from_docs(docs)
    return combined_docs

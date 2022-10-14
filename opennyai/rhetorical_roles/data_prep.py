from spacy.pipeline import Sentencizer
from typing import Optional, List, Callable
from spacy.language import Language
from spacy.tokens import Span
from typing import Optional, List, Callable

from spacy.language import Language
from spacy.pipeline import Sentencizer
from spacy.tokens import Span


@Language.factory(
    "my_sentencizer",
    assigns=["token.is_sent_start", "doc.sents"],
    default_config={"punct_chars": None, "overwrite": False, "scorer": {"@scorers": "spacy.senter_scorer.v1"}},
    default_score_weights={"sents_f": 1.0, "sents_p": 0.0, "sents_r": 0.0},
)
def make_sentencizer(
        nlp: Language,
    name: str,
    punct_chars: Optional[List[str]],
    overwrite: bool,
    scorer: Optional[Callable],):
    return mySentencizer(name, punct_chars=punct_chars, overwrite=overwrite, scorer=scorer)


class mySentencizer(Sentencizer):
    def predict(self, docs):
        """Apply the pipe to a batch of docs, without modifying them.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The predictions for each document.
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            guesses = [[] for doc in docs]
            return guesses
        guesses = []
        for doc in docs:
            doc_guesses = [False] * len(doc)
            if len(doc) > 0:
                start = 0
                seen_period = False
                doc_guesses[0] = True
                for i, token in enumerate(doc):
                    is_in_punct_chars = bool(re.match(r'^\n\s*$',token.text)) ####### hardcoded punctuations to newline characters
                    if seen_period and not is_in_punct_chars:
                        doc_guesses[start] = True
                        start = token.i
                        seen_period = False
                    elif is_in_punct_chars:
                        seen_period = True
                if start < len(doc):
                    doc_guesses[start] = True
            guesses.append(doc_guesses)
        return guesses

def get_spacy_nlp_pipeline_for_preamble(vocab=None,model_name="en_core_web_trf"):
    ########## Creates spacy nlp pipeline for Judgment Preamble. the sentence splitting is done on new lines.
    if vocab is not None:
        nlp = spacy.load(model_name,vocab=vocab,exclude=['ner'])
    else:
        nlp = spacy.load(model_name, exclude=['ner'])
    nlp.max_length = 30000000
    ########### Split sentences on new lines for preamble
    nlp.add_pipe("my_sentencizer", before='parser')
    return nlp


def extract_proper_nouns(sent,keywords):
    proper_nouns_list = []
    current_proper_noun_start = None
    for token in sent:
        if token.pos_=="PROPN" and token.lower_ not in keywords:
            if current_proper_noun_start is None:
                current_proper_noun_start = token.i
        elif current_proper_noun_start is not None and ((token.pos_ != 'ADP' and not token.is_punct) or token.lower_ in keywords):
            proper_nouns_list.append(sent.doc[current_proper_noun_start:token.i])
            current_proper_noun_start = None

    return proper_nouns_list

def match_span_with_keyword(span,keyword_dict):
    ########## matches the keywords in the given input span which is part of input sent
    span_label = None
    ##### check if court
    if span.text.lower().__contains__('court'):
        span_label ='COURT'
    else:
        ######check for judge patterns
        last_non_space_token = []
        if len([token for token in span if token.lower_ in keyword_dict['judge_keywords']]) > 0 or span.text.strip().endswith('J.'):
            span_label='JUDGE'
        else:
            ############# check for lawyer pattern
            if len([token for token in span if token.lower_ in keyword_dict['lawyer_keywords']]) > 0:
                span_label='LAWYER'
            else:
                ########## check for petitioner
                if len([token for token in span if token.lower_ in keyword_dict['petitioner_keywords']]) > 0:
                    span_label='PETITIONER'
                elif len([token for token in span if token.lower_ in keyword_dict['respondent_keywords']]) > 0:
                    span_label='RESPONDENT'
    return span_label


def validate_label(text_to_evaluate, sent_label):
    ########## checks to validate the chunk text
    valid_label= True
    if sent_label=='COURT' and not text_to_evaluate.lower().__contains__('court'):
        valid_label = False
    return valid_label


def add_chunk_entities(new_ents,block_ents,label_for_unknown_ents,doc,block_start_with_sequence_number,label_indicated_by_previous_block):
    sequence_number_suggested_next_block_label = None
    for block_ent in block_ents:
        entity_label = block_ent['label']
        if entity_label != 'UNKNOWN':
            final_entity_label = entity_label
        elif entity_label == 'UNKNOWN' and label_for_unknown_ents is not None:
            final_entity_label = label_for_unknown_ents
        else:
            final_entity_label = None

        if final_entity_label is not None:
            valid_label = validate_label(doc[block_ent['start']:block_ent['end']].text.lower(), final_entity_label)
            if valid_label:
                new_ent = Span(doc, block_ent['start'], block_ent['end'], label=final_entity_label)
                new_ents.append(new_ent)
                if final_entity_label in ['PETITIONER', 'RESPONDENT']:
                    if block_start_with_sequence_number and final_entity_label == label_indicated_by_previous_block:
                        sequence_number_suggested_next_block_label = final_entity_label
                    else:
                        sequence_number_suggested_next_block_label = None
                    ##### choose the first entity of the block for PETITIONER & RESPONDENT
                    break
    return sequence_number_suggested_next_block_label

def get_next_block_label(keyword_suggested_next_block_label,sequence_number_suggested_next_block_label):
    if keyword_suggested_next_block_label:
        next_block_label = keyword_suggested_next_block_label
    elif sequence_number_suggested_next_block_label:
        next_block_label = sequence_number_suggested_next_block_label
    else:
        next_block_label = None
    return next_block_label




def check_if_sentence_is_at_end_of_block(text):
    ########## check if sentence is ending with multiple new lines or the keywords that define end of block
    next_block_label = None
    current_block_end= False
    if re.match(r'^\s*Between\:?\s*$', text):
        next_block_label = "PETITIONER"
        current_block_end = True
    elif re.match(r'^\s*And\:?\s*$', text) or re.match(r'^\s*v\/?s[\:\s\.]*$', text,
                                                            re.IGNORECASE) or re.match(r'^\s*versus[\:\s\.]*$',
                                                                                       text, re.IGNORECASE):
        next_block_label = "RESPONDENT"
        current_block_end = True
    elif re.match(r'.*\n *\n+ *$', text):
        current_block_end = True

    return current_block_end ,next_block_label


def get_label_for_unknown_ents(block_label, label_indicated_by_previous_block):
    ######## for the entities where keywords are not found in same sentence, try to see if block labels could be used
    if block_label is not None:
        label_for_unknown_ents = block_label
    elif block_label is None and label_indicated_by_previous_block is not None:
        label_for_unknown_ents = label_indicated_by_previous_block
    else:
        label_for_unknown_ents = None
    return label_for_unknown_ents

@Language.component("extract_preamble_entities")
def extract_preamble_entities(doc):
    keyword_dict = {
    'lawyer_keywords' : ['advocate','adv.','counsel','lawyer','adv','advocates'],
    'judge_keywords' : ['justice','honourable',"hon'ble",'coram',"coram:","bench"],
    'petitioner_keywords' : ['appellant','petitioner','appellants','petitioners','petitioner(s)','petitioner(s','applicants','applicant','prosecution','complainant'],
    'respondent_keywords' : ['respondent','defendent','respondents'],
    'stopwords':['mr.','mrs.']}

    keywords = []
    for key,kw_list in keyword_dict.items():
        keywords.extend(kw_list)

    new_ents = []
    block_label = None
    next_block_label = None
    block_ents = []
    current_block_end =True
    block_start_with_sequence_number = False
    label_indicated_by_previous_block = None
    for sent in doc.sents:
        ###### check if new block is starting with serial number
        if current_block_end:
            if re.match(r'^\d[\.\)\]\s]+.*',sent.text):
                block_start_with_sequence_number = True
            current_block_end = False #### reset the block end flag

        ########## get the entity type by matching with keywords
        sent_label = match_span_with_keyword(sent, keyword_dict)

        ########## Use the first sentence label for the block label
        if sent_label is not None and block_label is None:
            block_label = sent_label

        ########### get proper nouns from sentence which are candidates for entities
        sent_proper_nouns = extract_proper_nouns(sent, keywords)


        for chunk in sent_proper_nouns:
            if sent_label is not None:
                ######## add proper nouns to entities where keywords are present in same sentence.
                new_ent = {'start':chunk.start, 'end':chunk.end, 'label' : sent_label}
                block_ents.append(new_ent)
            else:
                ###### decide the entity of proper noun later based on block keywords
                new_ent = {'start': chunk.start, 'end': chunk.end, 'label': 'UNKNOWN'}
                block_ents.append(new_ent)

        ######### Identify end of block
        current_block_end,keyword_suggested_next_block_label = check_if_sentence_is_at_end_of_block(sent.text)

        ######## if current block is ending then choose entities to be added
        if current_block_end:
            label_for_unknown_ents = get_label_for_unknown_ents(block_label,label_indicated_by_previous_block)
            sequence_number_suggested_next_block_label = add_chunk_entities(new_ents,block_ents,label_for_unknown_ents,doc,block_start_with_sequence_number,label_indicated_by_previous_block)
            next_block_label = get_next_block_label(keyword_suggested_next_block_label,sequence_number_suggested_next_block_label)

            block_ents= []
            block_start_with_sequence_number = False
            label_indicated_by_previous_block = next_block_label
            block_label = None


    doc.ents = new_ents
    return doc


import spacy
import re
from spacy.language import Language
from spacy.tokens import Span

def get_citation(doc, text, starts):
    '''Uses regex to identify citations in the judgmment and returns citation as a new entity'''
    regex = '(\(\d+\)|\d+|\[\d+\])\s*(\(\d+\)|\d+|\[\d+\])*\s*[A-Z]+\s*(\(\d+\)|\d+|\[\d+\])+\s*(\(\d+\)|\d+|\[\d+\])*\s*'
    new_ents = []
    for match in re.finditer(regex, text):
        token_number_start = starts.index(min(starts, key=lambda x: abs(match.span()[0] - x)))
        token_number_end = starts.index(min(starts, key=lambda x: abs(match.span()[1] - x)))
        if '(' in doc[token_number_start:token_number_end].text and ')' in doc[
                                                                           token_number_start:token_number_end].text:
            ent = Span(doc, token_number_start, token_number_end, label="CITATION")
            new_ents.append(ent)

    return new_ents



def get_police_station(doc, text, starts):
    '''Uses regex to identify the police station and returns PoliceStation as a new entity'''
    new_ents = []
    regex_ps = r'(?i)\bp\.*s\.*\b'

    for match in re.finditer(regex_ps, text):
        token_number = starts.index(min(starts, key=lambda x: abs(match.span()[0] - x)))
        i = token_number - 1

        while doc[i].text[0].isupper():
            token_number_start = i
            i = i - 1
        token_start = i + 1
        if token_start != token_number:
            ent = Span(doc, token_start, token_number + 1, label="PoliceStation")

            new_ents.append(ent)
    return new_ents


def get_precedents(doc, text, starts):
    '''Uses regex to identify the precedents based on keyword 'vs',merges citations with precedents and returns precedent as a new entity'''
    new_ents = []
    final_ents = []
    regex_vs = r'(?i)\sv\.*s*\.*\b'
    for match in re.finditer(regex_vs, text):
        token_number = starts.index(min(starts, key=lambda x: abs(match.span()[0] - x)))
        token_number_start = token_number
        token_number_end = token_number

        i = token_number_start - 1
        j = token_number_end + 1

        while doc[i].text[0].isupper() or doc[i].text.startswith('other') or doc[i].text in (
                'of', '-', '&', '@', '(', ')', '\n', '.') or doc[i].text.isdigit():
            if ',' in doc[i].text:
                break
            token_number_start = i
            i = i - 1

        while doc[j].text[0].isupper() or doc[j].text.startswith('other') or doc[j].text in (
                'of', '-', '&', '@', '(', ')', '\n', 'others', '.') or doc[i].text.isdigit():
            token_number_end = j

            if ',' in doc[j].text:
                break
            j = j + 1

        if token_number_end > token_number_start + 2 and token_number_start != token_number and token_number_end != token_number:
            ent = Span(doc, token_number_start, token_number_end + 1, label="PRECEDENT")

            final_ents.append(ent)

    citation_entities = get_citation(doc, text, starts)

    for ents in final_ents:
        token_num = ents.end
        if len(citation_entities) == 0:
            break
        citation_entity = (min(citation_entities, key=lambda x: abs(ents.end - x.start)))

        if (token_num + 1 == citation_entity.start or  token_num > citation_entity.start )and token_num < citation_entity.end:

            citation_entities.remove(citation_entity)
            ent = Span(doc, ents.start, citation_entity.end, label="PRECEDENT")
            new_ents.append(ent)
        else:
            new_ents.append(ents)

    for citation_entity in citation_entities:
        ent = Span(doc, citation_entity.start, citation_entity.end, label="PRECEDENT")
        new_ents.append(ent)

    for ents in final_ents:
        new_ents.append(ents)
    return new_ents


def get_court_case(doc, text, starts):
    '''Uses regex to identify the case numbers  and returns CASE_NUMBER as a new entity'''
    new_ents = []
    regex_court_case = r'((?i)(no.)+(\s*|\n)[0-9]+\s*(/|of)\s*[0-9]+)'

    for match in re.finditer(regex_court_case, text):

        token_number = starts.index(min(starts, key=lambda x: abs(match.span()[0] - x)))
        i = token_number - 1
        while doc[i].text[0].isupper():
            token_number = i
            i = i - 1
        start_char = starts[token_number]
        end_char = match.span()[1]

        ent = doc.char_span(start_char, end_char, label="CASE_NUMBER", alignment_mode="expand")
        new_ents.append(ent)
    return new_ents




def get_provisions(doc):
    '''Uses regex to identify the provision based on keyword section and returns Provision as a new entity'''
    new_ents = []

    for i, token in enumerate(doc):

        text = token.text.lower().strip()
        spans_start = -1
        spans_end = -1
        if text in ['section', 'sub-section', 'sections', 's.', 'ss.', 's', 'ss', 'u/s', 'u/s.', 'u/ss', 'u/s.s']:

            spans_start = i

            count = i + 1
            next_token = doc[count]
            next_text = next_token.text.strip().lower()

            while num_there(next_text) or next_text in ['to', 'and', ',', '/', '', '(', ')', '.']:
                count = count + 1
                next_token = doc[count]
                next_text = next_token.text.strip().lower()
            i = count - 1
            spans_end = i

        if spans_start != -1:
            if num_there(doc[spans_start:spans_end + 1].text):
                ent = Span(doc, spans_start, spans_end + 1, label="PROVISION")

                new_ents.append(ent)
    return new_ents


def filter_overlapping_entities(ents):
    '''Removes the overlapping entities in the judgmnent text'''
    filtered_ents = []
    for span in spacy.util.filter_spans(ents):
        filtered_ents.append(span)
    return filtered_ents

def get_entity(regex,doc,text,label):
    '''returns entity based on the given regex'''
    new_ents = []
    for x in re.finditer(regex, text):
        ent = doc.char_span(x.span()[0], x.span()[1], label=label, alignment_mode="expand")

        new_ents.append(ent)

    return new_ents


@Language.component("detect_pre_entities")
def detect_pre_entities(doc):
    '''Detects entities before ner using keyword matching'''
    text = doc.text

    starts = [tok.idx for tok in doc]
    new_ents = []
    final_ents = []


    regex_res = r'(?i)\b(respondent|respondents)\s*(((?i)no\.\s*\d+)|((?i)numbers)|((?i)number)|((?i)nos\.\s*\d+))*\s*(\d+|\,|and|to|\s*|–)+'
    regex_statute = r'(?i)((i\.*\s*p\.*\s*c\.*\s*)|(c\.*\s*r\.*\s*p\.*\s*c\.*\s*)|(indian*\s*penal\s*code\s*)|(penal\s*code\.*\s*)\n*)'
    regex_pw = r"\b(((?i)\s*\(*(P\.*W\.*s*)+\-*\s*(\d*\s*\,*\)*(and|to)*)*)|(?i)witness\s*)"
    regex_app = r'(?i)\b(appellant|appellants)\s*(((?i)no\.\s*\d+)|((?i)numbers)|((?i)number)|((?i)nos\.\s*\d+))*\s*(\d+|\,|and|to|\s*|–)+'
    respondent_keywords = get_entity(regex_res,doc,text,'key-rs')
    appellant_keywords = get_entity(regex_app,doc,text,'key-ap')
    witness_keywords =get_entity(regex_pw,doc,text,'key-pw')
    police_station = get_police_station(doc, text, starts)
    precedents = get_precedents(doc, text, starts)
    court_cases = get_court_case(doc, text, starts)
    statutes = get_entity(regex_statute,doc,text,'key-pw')
    provisions = get_provisions(doc)
    new_ents.extend(respondent_keywords)
    new_ents.extend(appellant_keywords)
    new_ents.extend(witness_keywords)
    new_ents.extend(police_station)
    new_ents.extend(precedents)
    new_ents.extend(court_cases)
    new_ents.extend(statutes)
    new_ents.extend(provisions)

    new_ents = filter_overlapping_entities(new_ents)

    doc.ents = new_ents

    return doc


def num_there(s):
    '''checks if string contains a digit'''
    return any(i.isdigit() for i in s)


def get_provision_statute_from_law_using_of(doc, ent):
    '''Detects provision and statute from entity law identified by default NER by breaking on keyword 'of'''
    new_ents = []
    ent_text = ent.text
    if ent_text.lower().find('section') > -1:
        section = ent_text.lower().find('section')
    elif ent_text.lower().find('sub-section') > -1:
        section = ent_text.lower().find('sub-section')
    else:
        section = -1

    if section != -1:
        if section < ent_text.find('of'):
            ent = doc.char_span(ent.start_char, ent.start_char + ent_text.find('of'), label="PROVISION",
                                alignment_mode="expand")
            new_ents.append(ent)
            ent = doc.char_span(ent.start_char + ent_text.find('of') + 2, ent.end_char, label="STATUTE",
                                alignment_mode="expand")
            new_ents.append(ent)
        else:
            ent = doc.char_span(ent.start_char + ent_text.find('of') + 2, ent.end_char, label="PROVISION",
                                alignment_mode="expand")
            new_ents.append(ent)
            ent = doc.char_span(ent.start_char, ent.start_char + ent_text.find('of'), label="STATUTE",
                                alignment_mode="expand")
            new_ents.append(ent)
    return new_ents


def get_provision_statute_from_law_using_keyword(doc, ent):
    '''Detects provision and statute from entity law identified by default NER' using keywords'''
    new_ents = []
    ent_text = ent.text
    if ent_text.lower().find('act') != -1:
        ent.label_ = 'STATUTE'
        new_ents.append(ent)
    elif ent_text.lower().find('section') != -1:
        ent.label_ = 'PROVISION'
        new_ents.append(ent)
    else:
        new_ents.append(ent)
    return new_ents


def get_prpopern_entitiy(doc, ent, entity_label):
    '''Detects the propernoun/person in the given string'''
    token_num = ent.end
    new_ents = []

    while len(doc) > token_num and (doc[token_num].ent_type_ == "PERSON" or doc[token_num].text == ',' or doc[token_num].pos_ == 'PROPN'):
        token_num = token_num + 1
    if token_num > ent.end + 1:
        new_ent = Span(doc, ent.end, token_num, label=entity_label )
        new_ents.append(new_ent)
    return new_ents


def get_witness(doc, new_ent):
    '''Detects witness using the keyword key-pw'''
    new_ents = []
    token_num_end = new_ent.end
    token_num_start = new_ent.start - 1

    while len(doc) > token_num_end and (
            doc[token_num_end].ent_type_ == "PERSON" or doc[token_num_end].text == ',' or doc[
        token_num_end].pos_ == 'PROPN'):
        token_num_end = token_num_end + 1

    while doc[token_num_start].ent_type_ == "PERSON" or doc[token_num_start].text == ',' or doc[
        token_num_end].pos_ == 'PROPN':
        token_num_start = token_num_start - 1

    if token_num_end > new_ent.end + 1:
        ent = Span(doc, new_ent.end, token_num_end, label="WITNESS")
        new_ents.append(ent)

    if token_num_start < new_ent.start and doc[token_num_start + 1].text != ',':
        ent = Span(doc, token_num_start + 1, new_ent.start, label="WITNESS")

        new_ents.append(ent)
    return new_ents


@Language.component("detect_post_entities")
def detect_post_entities(doc):
    '''Works on top of default NER to identify entities'''
    new_ents = []

    for new_ent in list(doc.ents):

        ent_text = new_ent.text

        if new_ent.label_ == "LAW":

            if 'case no.' in ent_text.lower():
                new_ent.label_ = 'CASE_NUMBER'
                new_ents.append(new_ent)

            elif ent_text.find('of') != -1:
                provision_statute_entities = get_provision_statute_from_law_using_of(doc, new_ent)
                new_ents.extend(provision_statute_entities)
            else:
                provision_statute_entities = get_provision_statute_from_law_using_keyword(doc, new_ent)
                new_ents.extend(provision_statute_entities)
        elif new_ent.label_ == "ORG":
            if 'court' in ent_text.lower():
                if len(ent_text.split(' ')) > 1:
                    ent = doc.char_span(new_ent.start_char, new_ent.end_char, label="COURT", alignment_mode="expand")
                    new_ents.append(ent)

            elif 'police station' in ent_text.lower():
                token_num = new_ent.end
                while len(doc) > token_num and (doc[token_num].ent_type_ == "GPE" or doc[token_num].text == ','):
                    token_num = token_num + 1
                if token_num > new_ent.end:
                    ent = Span(doc, new_ent.start, token_num, label="POLICE STATION")
                    new_ents.append(ent)
            else:

                token_num = new_ent.end
                while len(doc) > token_num and (doc[token_num].ent_type_ == "GPE" or doc[token_num].text == ','):
                    token_num = token_num + 1

                ent = Span(doc, new_ent.start, token_num, label="ORG")
                new_ents.append(ent)


        elif new_ent.label_ == "key-rs":
            respondents = get_prpopern_entitiy(doc, new_ent, 'RESPONDENT')
            new_ents.extend(respondents)
        elif new_ent.label_ == "key-ap":
            appellants = get_prpopern_entitiy(doc, new_ent, 'APPELLANT')
            new_ents.extend(appellants)
        elif new_ent.label_ == "key-pw":
            witness = get_witness(doc, new_ent)
            new_ents.extend(witness)


        else:
            new_ents.append(new_ent)

    new_ents = [ent for ent in new_ents if
                ent.label_ not in ['GPE', 'PERSON', 'LAW', 'DATE', 'MONEY', 'CARDINAL','ORDINAL','FAC']]

    new_ents = filter_overlapping_entities(new_ents)

    doc.ents = new_ents

    return doc

def get_judgment_text_pipeline():
    '''Returns the spacy pipeline for processing of the judgment text'''
    nlp_judgment = spacy.load("en_core_web_trf", disable=[])
    nlp_judgment.add_pipe("detect_pre_entities", before="ner")
    nlp_judgment.add_pipe("detect_post_entities", after="ner")
    return nlp_judgment

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
    max_length = 10000
    preamble_end = 0
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
    keyword_preamble_end_offset = remove_unwanted_text(txt)
    if keyword_preamble_end_offset == 0:
        preamble_end_offset = 5000  ######## if keywords not found then set arbitrarty value
    else:
        preamble_end_offset = keyword_preamble_end_offset + 200  ######## take few more characters as judge names are written after JUDGEMENT keywords
    truncated_txt = txt[:preamble_end_offset]
    guessed_preamble_end = guess_preamble_end(truncated_txt, preamble_splitting_nlp)

    if guessed_preamble_end == 0:
        preamble_end = keyword_preamble_end_offset
    else:
        preamble_end = guessed_preamble_end

    preamble_txt = txt[:preamble_end]
    # title_txt = convert_upper_case_to_title(preamble_txt)
    return preamble_txt, preamble_end


def get_spacy_nlp_pipeline_for_indian_legal_text(model_name="en_core_web_sm", disable=['ner'], punc=[".", "?", "!"],
                                                 custom_ner=False):
    ########## Creates spacy nlp pipeline for indian legal text. the sentence splitting is done on specific punctuation marks.
    #########This is finalized after multiple experiments comparison. To use all components pass empty list  disable = []

    import spacy
    from spacy.pipeline import Sentencizer
    try:
        spacy.prefer_gpu()
    except:
        pass
    nlp = spacy.load(model_name, disable=disable)
    nlp.max_length = 30000000

    ############ special tokens which should not be split in tokenization.
    #           this is specially helpful when we use models which split on the dots present in these abbreviations
    # special_tokens_patterns_list = [r'nos?\.',r'v\/?s\.?',r'rs\.',r'sh?ri\.']
    # special_tokens = re.compile( '|'.join(special_tokens_patterns_list),re.IGNORECASE).match
    # nlp.tokenizer = Tokenizer(nlp.vocab,token_match = special_tokens)

    ############## Custom NER patterns
    patterns = [{"label": "RESPONDENT",
                 "pattern": [{"LOWER": "respondent"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(\d+|\,|and|to)"}, "OP": "+"},
                             {"TEXT": {"REGEX": "\d+"}}]},
                {"label": "RESPONDENT",
                 "pattern": [{"LOWER": "respondent"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(^(?i)no\.\d+$)"}, "OP": "*"},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"TEXT": {"REGEX": r"^(?i)PW\-\d*\w+$"}},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"LOWER": "prosecution"}, {"TEXT": {"REGEX": r"(^(?i)Witness\-\S+$)|(^(?i)witness)"}},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "WITNESS",
                 "pattern": [{"TEXT": {"REGEX": "(^(?i)PW$)"}}, {"TEXT": {"REGEX": "(\d+)"}},
                             {"TEXT": {"REGEX": r"^(\/|[A-Z])$"}, "OP": "*"}]},
                {"label": "ACCUSED",
                 "pattern": [{"LOWER": "accused"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(\d+|\,|and|to)"}, "OP": "+"},
                             {"TEXT": {"REGEX": "\d+"}}]},
                {"label": "ACCUSED",
                 "pattern": [{"LOWER": "accused"},
                             {"TEXT": {"REGEX": "(^(?i)numbers$)|(^(?i)number$)|(^(?i)nos\.\d+$)|(^(?i)no\.\d+$)"}},
                             {"TEXT": {"REGEX": "(^(?i)no\.\d+$)"}, "OP": "*"},
                             {"TEXT": {"REGEX": "(\d+)"}, "OP": "*"}]}]

    if int(spacy.__version__.split(".")[0]) >= 3:
        ########### For transformer model use built in sentence splitting. For others, use sentence splitting on punctuations.
        ########### This is because transformer sentence spiltting is doing better than the punctuation spiltting
        if model_name != "en_core_web_trf":
            config = {"punct_chars": punc}
            nlp.add_pipe("sentencizer", config=config, before='parser')

        if "ner" not in disable and custom_ner:
            ruler = nlp.add_pipe("entity_ruler", before='ner')
            ruler.add_patterns(patterns)

    else:
        if model_name != "en_core_web_trf":
            sentencizer = Sentencizer(punct_chars=punc)
            nlp.add_pipe(sentencizer, before="parser")
        if "ner" not in disable and custom_ner:
            from spacy.pipeline import EntityRuler
            ruler = EntityRuler(nlp, overwrite_ents=True)
            ruler.add_patterns(patterns)
            nlp.add_pipe(ruler, before='ner')

    return nlp


def attach_short_sentence_boundries_to_next(revised_sentence_boundries, doc_txt):
    ###### this function accepts the list in the format of output of function "extract_relevant_sentences_for_rhetorical_roles" and returns the revised list with shorter sentences attached to next sentence
    min_char_cnt_per_sentence = 5

    concatenated_sentence_boundries = []
    sentences_to_attach_to_next = ()
    for sentence_boundry in revised_sentence_boundries:
        sentence_txt = doc_txt[sentence_boundry[0]: sentence_boundry[1]]
        if not sentence_txt.isspace():  ### sentences containing only spaces , newlines are discarded
            if sentences_to_attach_to_next:
                sentence_start_char = sentences_to_attach_to_next[0]
            else:
                sentence_start_char = sentence_boundry[0]
            # sentence_length_char = sentence_boundry[1] - sentence_start_char
            sentence_length_char = len(doc_txt[sentence_start_char: sentence_boundry[1]].strip())
            if sentence_length_char > min_char_cnt_per_sentence:
                concatenated_sentence_boundries.append((sentence_start_char, sentence_boundry[1]))
                sentences_to_attach_to_next = ()
            else:
                if not sentences_to_attach_to_next:
                    sentences_to_attach_to_next = sentence_boundry
    return concatenated_sentence_boundries


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


def split_preamble_judgement(judgment_txt):
    ###### seperates the preamble and judgement text for all courts. It removes the new lines in between  the sentences.  returns 2 texts
    preamble_end = remove_unwanted_text(judgment_txt)
    preamble_removed_txt = judgment_txt[preamble_end:]
    preamble_txt = judgment_txt[:preamble_end]

    ####### remove the new lines which are not after dot or ?. Assumption is that theses would be in between sentence
    preamble_removed_txt = re.sub(r'([^.\"\?])\n+ *', r'\1 ',
                                  preamble_removed_txt)
    return preamble_txt, preamble_removed_txt

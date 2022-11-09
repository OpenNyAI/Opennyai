import json
import os

import spacy
from tqdm import tqdm
from wasabi import msg

spacy.prefer_gpu()


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


def split_into_sentences_tokenize_write(data, custom_processed_data_path,
                                        hsln_format_txt_dirpath='datasets/pubmed-20k', verbose=False):
    ########## This function accepts the input files in LS format, creates tokens and writes them with label as "NONE" to text file

    if not os.path.exists(hsln_format_txt_dirpath):
        os.makedirs(hsln_format_txt_dirpath)
    max_length = 10000
    output_json = []
    filename_sent_boundries = {}  ###### key is the filename and value is dict containing sentence spans {"abc.txt":{"sentence_span":[(1,10),(11,20),...]} , "pqr.txt":{...},...}
    if verbose:
        msg.info('Preprocessing rhetorical role model input!!!')
    for data_dict in tqdm(data, disable=not verbose):

        doc_id = data_dict['file_id']
        preamble_doc = data_dict['preamble_doc']
        judgment_doc = data_dict['judgement_doc']

        if filename_sent_boundries.get(doc_id) is None:  ##### Ignore if the file is already present

            nlp_doc = spacy.tokens.Doc.from_docs([preamble_doc, judgment_doc])
            doc_txt = nlp_doc.text
            sentence_boundries = [(sent.start_char, sent.end_char) for sent in nlp_doc.sents]
            revised_sentence_boundries = attach_short_sentence_boundries_to_next(sentence_boundries, doc_txt)

            adjudicated_doc = {'id': doc_id,
                               'data': {'preamble_text': preamble_doc.text,
                                        'judgement_text': judgment_doc.text,
                                        'text': doc_txt}
                               }

            adjudicated_doc['annotations'] = []
            adjudicated_doc['annotations'].append({})
            adjudicated_doc['annotations'] = []

            filename_sent_boundries[doc_id] = {"sentence_span": []}
            for sentence_boundry in revised_sentence_boundries:
                sentence_txt = doc_txt[sentence_boundry[0]:sentence_boundry[1]]

                if sentence_txt.strip() != "":
                    sentence_txt = sentence_txt.replace("\r", "")
                    sent_data = {}
                    sent_data['start'] = sentence_boundry[0]
                    sent_data['end'] = sentence_boundry[1]
                    sent_data['text'] = sentence_txt
                    sent_data['labels'] = []
                    adjudicated_doc['annotations'].append(sent_data)

        output_json.append(adjudicated_doc)
    with open(custom_processed_data_path, 'w+') as f:
        json.dump(output_json, f)


def write_in_hsln_format(input_json, hsln_format_txt_dirpath, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    json_format = json.load(open(input_json))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name = file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations']:
            filename_sent_boundries[file_name]['sentence_span'].append(
                [annotation['start'], annotation['end']])

            sentence_txt = annotation['text']
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + "NONE" + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + '/test_scibert.txt', "w+") as file:
        file.write(final_string)

    with open(hsln_format_txt_dirpath + '/train_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/dev_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/sentece_boundries.json', 'w+') as json_file:
        json.dump(filename_sent_boundries, json_file)

    return filename_sent_boundries

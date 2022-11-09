import copy

from ..prepro.data_builder import BertData, greedy_selection


def preprocess_for_summarization(input_data: dict, bert_tokenizer, tokenizer):
    """
    input_data={"id":file_name,data{"text"},"annotations":[....]}
    :param bert_tokenizer:
    :param tokenizer:
    :param input_data:
    :return bert_preprocessed_data:
    """
    max_bert_tokens_per_chunk = 510

    source_chunk_id = 0
    doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [], 'source_filename': input_data['id'],
                'sentence_id': [], 'src_chunk_id': source_chunk_id}
    doc_data_list = []

    for index, value in enumerate(input_data['annotations']):
        value = value
        if not value['labels'][0] in ['NONE', 'PREAMBLE']:
            tokenized = bert_tokenizer.tokenize(value['text'])
            sent_tokens = [token.text for token in tokenizer(value['text'])]
            if (sum([len(bert_tokenizer.tokenize(' '.join(i))) + 2 for i in doc_data['src']]) + len(
                    tokenized)) <= max_bert_tokens_per_chunk:
                doc_data['src'].append(sent_tokens)
                doc_data['src_rhetorical_roles'].append(value['labels'][0])
                doc_data['sentence_id'].append(value['id'])
            elif len(
                    tokenized) > max_bert_tokens_per_chunk:
                if doc_data['src']:
                    doc_data_list.append(copy.deepcopy(doc_data))
                    source_chunk_id += 1
                    doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                'source_filename': input_data['id'],
                                'sentence_id': [],
                                'src_chunk_id': source_chunk_id}
                tokens_list = [tokenized[i:i + max_bert_tokens_per_chunk] for i in
                               range(0, len(tokenized), max_bert_tokens_per_chunk - 0)]
                if len(tokens_list[-1]) < 100:
                    tokens_list = tokens_list[:-1]
                misc_sentence_id = float(value['id'].split('_')[-1])
                for _ in tokens_list:
                    misc_sentence_id += 0.01
                    sent_tokens = bert_tokenizer.convert_tokens_to_string(_).split(
                        ' ')  # [token.text for token in self.tokenizer(value['text'])]
                    doc_data['src'].append(sent_tokens)
                    doc_data['src_rhetorical_roles'].append(value['labels'][0])
                    doc_data['sentence_id'].append(misc_sentence_id)
                    doc_data_list.append(copy.deepcopy(doc_data))
                    source_chunk_id += 1
                    doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                                'source_filename': input_data['id'],
                                'sentence_id': [],
                                'src_chunk_id': source_chunk_id}
            else:
                doc_data_list.append(copy.deepcopy(doc_data))
                source_chunk_id += 1
                doc_data = {"src": [], 'tgt': [], "src_rhetorical_roles": [],
                            'source_filename': input_data['id'],
                            'sentence_id': [],
                            'src_chunk_id': source_chunk_id}
                doc_data['src'].append(sent_tokens)
                doc_data['src_rhetorical_roles'].append(value['labels'][0])
                doc_data['sentence_id'].append(value['id'])

    if doc_data['src']:
        doc_data_list.append(copy.deepcopy(doc_data))

    return doc_data_list


def format_to_bert(bert_preprocessed_data, preprocessing_args):
    bert = BertData(preprocessing_args)
    datasets = []
    for d in bert_preprocessed_data:
        source, tgt = d['src'], d['tgt']
        f_name_chunk_id = '___'.join([d['source_filename'], str(d['src_chunk_id']), str(d['sentence_id'])])
        sent_labels = greedy_selection(source[:preprocessing_args.max_src_nsents], tgt, len(tgt))
        sent_rhetorical_roles = d['src_rhetorical_roles']
        if (preprocessing_args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, sent_rhetorical_roles,
                                 use_bert_basic_tokenizer=preprocessing_args.use_bert_basic_tokenizer,
                                 is_test=True)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, sent_rr, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "sentence_rhetorical_roles": sent_rr, "segs": segments_ids,
                       'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, 'unique_id': f_name_chunk_id}
        datasets.append(b_data_dict)
    return datasets

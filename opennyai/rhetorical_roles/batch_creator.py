import numpy as np
import torch

import opennyai.rhetorical_roles.bucketing as bucketing
from opennyai.rhetorical_roles.allennlp_helper.common.util import pad_sequence_to_length


class BatchCreator:
    def __init__(self, dataset, tokenizer, labels, batch_sentence_size, max_seq_length):
        # '''dataset: Iterable over documents
        #   tokenizer: WordPiece tokenizer of BERT. If None, then it is assumed that sentences are already tokenized.
        #   labels: possible lables of the sentences
        #   max_sequence_length: Max number of tokens for each sentences. Only required if tokenizer is provided.
        # '''
        self.dataset = dataset
        self.labels = labels
        self.batch_sentence_size = batch_sentence_size
        self.max_sequence_length = max_seq_length
        self.tokenizer = tokenizer
        self.batches = None

    def get_batches(self, task_name=None):
        self.build_batches()
        result = []
        for b in self.batches:
            batch = self.batch_to_tensor(b)
            if task_name is not None:
                batch["task"] = task_name
            result.append(batch)
        return result

    def build_batches(self):
        if self.batches is None:
            def wrap_document(doc):
                return bucketing.Record(doc.get_sentence_count(), doc)

            mapped_ds = map(wrap_document, self.dataset)
            self.batches = bucketing.bucket_records(mapped_ds, self.batch_sentence_size)

        return len(self.batches)

    def get_batches_count(self):
        batches_count = self.build_batches()
        return batches_count

    def batch_to_tensor(self, b):
        # dictionary of arrays
        tensors_dict_arrays = b.to_tensor(self.document_to_sequence_example, merge_records)
        # convert to dictionary of tensors and pad the tensors
        result = {}
        for k, v in tensors_dict_arrays.items():

            if k in ["input_ids", "attention_mask"]:
                # determine the max sentence len in the batch
                max_sentence_len = -1
                for doc in v:
                    for sentence in doc:
                        max_sentence_len = max(len(sentence), max_sentence_len)
                # pad the sentences to max sentence len
                for doc in v:
                    for i, sentence in enumerate(doc):
                        doc[i] = pad_sequence_to_length(sentence, desired_length=max_sentence_len)
            if k != 'doc_name':
                result[k] = torch.tensor(v)
            else:
                result[k] = v
        return result

    def document_to_sequence_example(self, document, sentence_padding_len):

        sentences = list(document.data.sentences)
        labels = list(document.data.labels)

        # pad number of sentences
        for _ in range(len(document.data.labels), sentence_padding_len):
            sentences.append("")
            labels.append("mask")

        token_ids = []
        attention_masks = []
        label_ids = []
        for sentence, label in zip(sentences, labels):
            if self.tokenizer is None:
                # sentence already tokenized
                if isinstance(sentence, list):
                    tok_ids = sentence
                else:
                    tok_ids = [int(t) for t in sentence.split()]
            else:
                tok_ids = self.tokenizer.encode(sentence, add_special_tokens=True, max_length=self.max_sequence_length)

            attention_mask = [1] * len(tok_ids)

            # map label id
            label_id = self.labels.index(label)

            token_ids.append(tok_ids)
            attention_masks.append(attention_mask)
            label_ids.append(label_id)

        return {
            "sentence_mask": pad_sequence_to_length([1] * document.length, desired_length=sentence_padding_len),
            "input_ids": token_ids,
            "attention_mask": attention_masks,
            "label_ids": label_ids,
            "doc_name": document.data.doc_name
        }


def merge_records(merged, r):
    if merged is None:
        merged = dict()
        for k in r:
            merged[k] = []

    for k in r:
        merged[k].append(r[k])

    return merged


def one_hot(num, v):
    r = np.zeros(num, dtype=int)
    r[v] = 1
    return r

import ast
import copy

import numpy as np
import torch
from spacy.lang.en import English
from tqdm import tqdm
from transformers import BertTokenizer
from wasabi import msg

from opennyai.utils.download import load_model_from_cache
from .models import data_loader
from .models.model_builder import ExtSummarizer
from .others.args import __setargs__
from .others.postprocessing_utils import _postprocess
from .others.utils import preprocess_for_summarization, format_to_bert


class ExtractiveSummarizer:
    def __init__(self, use_gpu: bool = True, verbose: bool = False, summary_length: float = 0.0):
        """Returns object of InLegalNER class.
         It is used for loading Extractive Summarizer model in memory.
        Args:
            use_gpu (bool): Functionality to give a choice whether to use GPU for inference or not
             Setting it True doesn't ensure GPU will be utilized it need proper torch installation
            verbose (bool): When set to True will print info msg while inference
            summary_length (float): valid range(0-1) Length of summary to get in output. set it to 0 to use adaptive selection
        """
        self.__verbose__ = verbose
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                msg.info('Extractive Summarizer will use GPU!')
            else:
                self.device = torch.device('cpu')
                msg.info('Extractive Summarizer will use CPU!')
        else:
            self.device = torch.device('cpu')
            msg.info('Extractive Summarizer will use CPU!')

        # load summarizer checkpoint
        state_dict = load_model_from_cache('ExtractiveSummarizer')

        # setup tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.__tokenizer__ = English().tokenizer

        # setup model arguments
        self.model_args, self.preprocessing_args = __setargs__()

        # setup model
        self.model = ExtSummarizer(self.model_args, self.device, state_dict)
        self.model.eval()
        if summary_length < 0 or summary_length > 1:
            summary_length_percentage = 0.0
            msg.info('Invalid input: Summary length need to be in range of 0-1. Setting it to adaptive')
        else:
            summary_length_percentage = float(summary_length)
        self.__summary_length_percentage__ = summary_length_percentage

    def _preprocess(self, input_data):
        try:
            if not input_data['annotations']:
                raise TypeError("Missing data in input for processing")
        except:
            raise TypeError("Invalid data format")
        bert_formatted = preprocess_for_summarization(input_data, self.bert_tokenizer, self.__tokenizer__)
        bert_formatted_and_tokenized = format_to_bert(bert_formatted, self.preprocessing_args)
        return bert_formatted_and_tokenized

    @staticmethod
    def _load_dataset(preprocessed_data):
        yield preprocessed_data

    def _inference(self, data_iter):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        self.model.eval()
        file_chunk_sent_scores = {}  ## key is filename and value is list of sentences containing sentence scores

        with torch.no_grad():
            for batch in data_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sentence_rhetorical_roles = batch.sentence_rhetorical_roles
                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls, sentence_rhetorical_roles)
                sent_scores = sent_scores.cpu().data.numpy()
                file_name, chunk_id, sentence_ids = batch.unique_ids[0].split('___')
                chunk_id = int(chunk_id)
                sentence_ids = ast.literal_eval(sentence_ids)
                src_labels = list(labels.cpu().numpy()[0])
                if type(sent_scores[0]) == np.float32:
                    sent_scores = np.array([sent_scores])
                sent_scores_list = list(sent_scores[0])
                sent_rhetorical_roles_list = list(sentence_rhetorical_roles.cpu().data.numpy()[0])
                for sent_id, (sent_txt, sent_label, sent_score, sent_rhet_role, sentence_id) in enumerate(
                        zip(batch.src_str[0], src_labels, sent_scores_list, sent_rhetorical_roles_list, sentence_ids)):
                    if file_chunk_sent_scores.get(file_name) is None:
                        file_chunk_sent_scores[file_name] = []
                    sent_dict = {'file_name': file_name, 'chunk_id': chunk_id, 'sent_id': sentence_id,
                                 'sent_txt': sent_txt,
                                 'sent_score': sent_score, 'sent_label': sent_label,
                                 'sent_rhetorical_role': sent_rhet_role}
                    file_chunk_sent_scores[file_name].append(sent_dict)

        return file_chunk_sent_scores

    def __call__(self, input_data):
        if self.__verbose__:
            msg.info('Processing documents with extractive summarizer model!!!')
        result = []
        for data in tqdm(input_data, disable=not self.__verbose__):
            task_id = data['id'].split('_')[-1]
            preprocessed_data = self._preprocess(data)
            data_iter = data_loader.Dataloader(self.model_args, self._load_dataset(preprocessed_data),
                                               self.model_args.test_batch_size, self.device,
                                               shuffle=False, is_test=True)
            inference_output = self._inference(data_iter)
            summary_texts = _postprocess(inference_output, data['annotations'], self.__summary_length_percentage__)
            result.append(copy.deepcopy({"id": 'ExtractiveSummarizer_' + task_id, "summaries": summary_texts}))
        return result

import json
import os
import shutil
import warnings
from pathlib import Path

import torch
from transformers import BertTokenizer
from wasabi import msg

import opennyai.rhetorical_roles.models as models
from opennyai.rhetorical_roles.infer_data_prep import split_into_sentences_tokenize_write, write_in_hsln_format
from opennyai.utils.download import load_model_from_cache
from .eval import infer_model
from .models import BertHSLN
from .task import pubmed_task


class RhetoricalRolePredictor():

    def __init__(self, use_gpu=True, verbose=False):
        BERT_MODEL = "bert-base-uncased"
        self.config = {
            "bert_model": BERT_MODEL,
            "bert_trainable": False,
            "model": BertHSLN.__name__,
            "cacheable_tasks": [],

            "dropout": 0.5,
            "word_lstm_hs": 758,
            "att_pooling_dim_ctx": 200,
            "att_pooling_num_ctx": 15,

            "lr": 3e-05,
            "lr_epoch_decay": 0.9,
            "batch_size": 1,
            "max_seq_length": 128,
            "max_epochs": 40,
            "early_stopping": 5,

        }
        self.use_gpu = use_gpu
        self.__verbose__ = verbose
        self.initialize()

    def initialize(self):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.CACHE_DIR = os.path.join(str(Path.home()), '.opennyai')
        self.hsln_format_txt_dirpath = os.path.join(self.CACHE_DIR, 'temp_hsln/pubmed-20k', )
        os.makedirs(self.hsln_format_txt_dirpath, exist_ok=True)

        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                msg.info('Rhetorical Roles will use GPU!')
            else:
                self.device = torch.device('cpu')
                msg.info('Rhetorical Roles will use CPU!')
        else:
            self.device = torch.device('cpu')
            msg.info('Rhetorical Roles will use CPU!')

        # Load model
        def create_task(create_func):
            return create_func(train_batch_size=self.config["batch_size"], max_docs=-1, data_folder=self.CACHE_DIR,
                               verbose=self.__verbose__)

        task = create_task(pubmed_task)
        self.model = getattr(models, self.config["model"])(self.config, [task])

        self.model.load_state_dict(load_model_from_cache('RhetoricalRole'))
        self.model.to(self.device)

        # Ensure to use the same tokenizer used during training
        BERT_VOCAB = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Tokenize the texts and write files
            split_into_sentences_tokenize_write(data, os.path.join(self.hsln_format_txt_dirpath, 'input_to_hsln.json'),
                                                hsln_format_txt_dirpath=self.hsln_format_txt_dirpath,
                                                verbose=self.__verbose__)

            write_in_hsln_format(os.path.join(self.hsln_format_txt_dirpath, 'input_to_hsln.json'),
                                 self.hsln_format_txt_dirpath, self.tokenizer)

            task = pubmed_task(train_batch_size=self.config["batch_size"], max_docs=-1,
                               data_folder=self.hsln_format_txt_dirpath, verbose=self.__verbose__)
            return task

    def __call__(self, data):
        """ Predict the class of a text using a trained transformer model.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            task = self.preprocess(data)
            folds = task.get_folds()
            test_batches = folds[0].test
            # metrics, confusion, labels_dict, class_report = eval_model(self.model, test_batches, self.device, task)
            labels_dict = infer_model(self.model, test_batches, self.device, task, verbose=self.__verbose__)
            filename_sent_boundries = json.load(
                open(os.path.join(self.hsln_format_txt_dirpath, 'sentece_boundries.json')))

            for doc_name, predicted_labels in zip(labels_dict['doc_names'], labels_dict['docwise_y_predicted']):
                filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels

            with open(os.path.join(self.hsln_format_txt_dirpath, 'input_to_hsln.json'), 'r') as f:
                input = json.load(f)
            for file in input:
                id = str(file['id'])
                pred_id = labels_dict['doc_names'].index(id)
                pred_labels = labels_dict['docwise_y_predicted']
                annotations = file['annotations']
                for i, label in enumerate(annotations):
                    import uuid
                    uid = uuid.uuid4()
                    label_id = uid.hex + '_' + str(i)
                    label['labels'] = [pred_labels[pred_id][i]]
                    label['id'] = label_id

                file['id'] = "RhetoricalRole_" + str(file['id'])

            shutil.rmtree(self.hsln_format_txt_dirpath)  ##### remove the temporary files
            return input

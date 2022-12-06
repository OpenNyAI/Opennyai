import copy
import math

import torch
from transformers import BertModel

from opennyai.rhetorical_roles.allennlp_helper.common.util import pad_sequence_to_length
from opennyai.rhetorical_roles.allennlp_helper.modules.conditional_random_field.conditional_random_field import \
    ConditionalRandomField
from opennyai.rhetorical_roles.allennlp_helper.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import \
    PytorchSeq2SeqWrapper
from opennyai.rhetorical_roles.allennlp_helper.nn.util import masked_softmax


class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''

    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label

            # log_denominator = self.crf._input_likelihood(logits, mask)
            # log_numerator = self.crf._joint_likelihood(logits, predicted_label, mask)
            # log_likelihood = log_numerator - log_denominator
            # outputs["log_likelihood"] = log_likelihood

        return outputs


class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''

    def __init__(self, in_dim, tasks):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for task in tasks:
            self.per_task_output[task.task_name] = CRFOutputLayer(in_dim=in_dim, num_labels=len(task.labels))

    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.per_task_output[task](x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, task in enumerate(self.per_task_output.keys()):
            if index % 2 == 0:
                self.task_to_device[task] = device1
                self.per_task_output[task].to(device1)
            else:
                self.task_to_device[task] = device2
                self.per_task_output[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]


class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        # shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s


class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        # state_dict_1 = self.bert.state_dict()
        # state_dict_2 = torch.load('/home/astha_agarwal/model/pytorch_model.bin')
        # for name2 in state_dict_2.keys():
        #    for name1 in state_dict_1.keys():
        #        temp_name = copy.deepcopy(name2)
        #       if temp_name.replace("bert.", '') == name1:
        #            state_dict_1[name1] = state_dict_2[name2]

        # self.bert.load_state_dict(state_dict_1,strict=False)

        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)

        # input_ids = batch["input_ids"].view(-1, tokens)
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # # shape (documents*sentences, tokens, 768)
        # bert_embeddings = outputs[0]

        #### break the large judgements into sentences chunk of given size. Do this while inference
        chunk_size = 1024
        input_ids = batch["input_ids"].view(-1, tokens)
        chunk_cnt = int(math.ceil(input_ids.shape[0] / chunk_size))
        input_ids_chunk_list = torch.chunk(input_ids, chunk_cnt)

        attention_mask_chunk_list = torch.chunk(attention_mask, chunk_cnt)
        outputs = []
        for input_ids, attention_mask in zip(input_ids_chunk_list, attention_mask_chunk_list):
            with torch.no_grad():
                output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                output = output[0]
                # output = output[0].to('cpu')
            outputs.append(copy.deepcopy(output))
            torch.cuda.empty_cache()

        bert_embeddings = torch.cat(tuple(outputs))  # .to('cuda')

        if not self.bert_trainable and batch["task"] in self.cacheable_tasks:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings


class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''

    def __init__(self, config, tasks):
        super(BertHSLN, self).__init__()

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config, tasks)

        self.reinit_output_layer(tasks, config)

    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        # print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                                                 hidden_size=self.lstm_hidden_size,
                                                                 num_layers=1, batch_first=True, bidirectional=True))

    def reinit_output_layer(self, tasks, config):
        if config.get("without_context_enriching_transfer"):
            self.init_sentence_enriching(config, tasks)
        input_dim = self.lstm_hidden_size * 2

        if self.generic_output_layer:
            self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
        else:
            self.crf = CRFPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        if self.generic_output_layer:
            output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)

        return output


class BertHSLNMultiSeparateLayers(torch.nn.Module):
    '''
    Model Multi-Task Learning, where only certail layers are shared.
    This class is necessary to separate the model on two GPUs.
    '''

    def __init__(self, config, tasks):
        super(BertHSLNMultiSeparateLayers, self).__init__()

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = PerTaskGroupWrapper(
            task_groups=config["attention_groups"],
            create_module_func=lambda g:
            AttentionPooling(2 * self.lstm_hidden_size,
                             dimension_context_vector_u=config["att_pooling_dim_ctx"],
                             number_context_vectors=config["att_pooling_num_ctx"])
        )

        attention_pooling_output_dim = next(iter(self.attention_pooling.per_task_mod.values())).output_dim
        self.sentence_lstm = PerTaskGroupWrapper(
            task_groups=config["context_enriching_groups"],
            create_module_func=lambda g:
            PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=attention_pooling_output_dim,
                                                hidden_size=self.lstm_hidden_size,
                                                num_layers=1, batch_first=True, bidirectional=True))
        )

        self.crf = CRFPerTaskGroupOutputLayer(self.lstm_hidden_size * 2, tasks, config["output_groups"])

    def to_device(self, device1, device2):
        self.bert.to(device1)
        self.word_lstm.to(device1)
        self.attention_pooling.to_device(device1, device2)
        self.sentence_lstm.to_device(device1, device2)
        self.crf.to_device(device1, device2)
        self.device1 = device1
        self.device2 = device2

    def forward(self, batch, labels=None, output_all_tasks=False):
        task_name = batch["task"]
        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        device = self.attention_pooling.get_device(task_name)
        sentence_embeddings = self.attention_pooling(task_name, bert_embeddings_encoded.to(device),
                                                     tokens_mask.to(device))
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        sentence_mask = batch["sentence_mask"]
        # shape: (documents, sentence, 2*lstm_hidden_size)
        device = self.sentence_lstm.get_device(task_name)
        sentence_embeddings_encoded = self.sentence_lstm(task_name, sentence_embeddings.to(device),
                                                         sentence_mask.to(device))
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        device = self.crf.get_device(task_name)
        if labels is not None:
            labels = labels.to(device)

        output = self.crf(task_name, sentence_embeddings_encoded.to(device), sentence_mask.to(device), labels,
                          output_all_tasks)

        return output


class CRFPerTaskGroupOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''

    def __init__(self, in_dim, tasks, task_groups):
        super(CRFPerTaskGroupOutputLayer, self).__init__()

        def get_task(name):
            for t in tasks:
                if t.task_name == name:
                    return t

        self.crf = PerTaskGroupWrapper(
            task_groups=task_groups,
            create_module_func=lambda g:
            # we assume same labels per group
            CRFOutputLayer(in_dim=in_dim, num_labels=len(get_task(g[0]).labels))
        )
        self.all_tasks = [t for t in [g for g in task_groups]]

    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.crf(task, x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for task in self.self.all_tasks:
                task_result = self.crf(task, x, mask, labels)
                task_result["task"] = task
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.crf.to_device(device1, device2)

    def get_device(self, task):
        return self.crf.get_device(task)


class PerTaskGroupWrapper(torch.nn.Module):
    def __init__(self, task_groups, create_module_func):
        super(PerTaskGroupWrapper, self).__init__()

        self.per_task_mod = torch.nn.ModuleDict()
        for g in task_groups:
            mod = create_module_func(g)
            for t in g:
                self.per_task_mod[t] = mod

        self.task_groups = task_groups

    def forward(self, task_name, *args):
        mod = self.per_task_mod[task_name]
        return mod(*args)

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, tasks in enumerate(self.task_groups):
            for task in tasks:
                if index % 2 == 0:
                    self.task_to_device[task] = device1
                    self.per_task_mod[task].to(device1)
                else:
                    self.task_to_device[task] = device2
                    self.per_task_mod[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]

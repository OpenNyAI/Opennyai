import torch
from wasabi import msg
from .others.args import __setargs__
from spacy.lang.en import English
from .models.model_builder import ExtSummarizer
from .others.tokenization import BertTokenizer
from opennyai.utils.download import load_model_from_cache


class ExtractiveSummarizer:
    def __init__(self, use_gpu: bool = True, verbose: bool = False):
        """Returns object of InLegalNER class.
         It is used for loading Extractive Summarizer model in memory.
        Args:
            use_gpu (bool): Functionality to give a choice whether to use GPU for inference or not
             Setting it True doesn't ensure GPU will be utilized it need proper torch installation
            verbose (bool): When set to True will print info msg while inference
        """

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

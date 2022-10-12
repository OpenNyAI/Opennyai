import spacy
from hashlib import sha256
from wasabi import msg
from tqdm import tqdm
from .download import models_url, install
from .sentencizer import split_main_judgement_to_preamble_and_judgement


class Data:
    def __init__(self, input_text, preprocessing_nlp_model='en_core_web_trf', mini_batch_size=40000, use_gpu=True,
                 use_cache=True, verbose=False):
        self._input_text = input_text
        self._mini_batch_size = mini_batch_size
        self._verbose = verbose
        self._use_cache = use_cache
        if self._use_cache:
            self._cache = {}
        if type(self._input_text) == str:
            self._input_text = [self._input_text]
        elif type(self._input_text) == list and len(self._input_text) >= 1:
            pass
        else:
            raise RuntimeError('No input or wrong given, we accept input as string or list of strings')
        if preprocessing_nlp_model not in spacy.util.get_installed_models():
            msg.info(f'Installing {preprocessing_nlp_model} this is a one time process!!')
            if models_url.get(preprocessing_nlp_model) is not None and preprocessing_nlp_model in ['en_core_web_trf',
                                                                                                   'en_core_web_sm',
                                                                                                   'en_core_web_md']:
                install(models_url[preprocessing_nlp_model])
            else:
                raise RuntimeError(
                    f'{preprocessing_nlp_model} doesn\'t exist in list of available opennyai preprocessing models')
        if use_gpu:
            try:
                if spacy.prefer_gpu():
                    msg.info(title='Pre-processing will happen on GPU')
                else:
                    msg.info(title='Pre-processing will happen on CPU')
                spacy.prefer_gpu()
            except:
                msg.info(title='Pre-processing will happen on CPU')
        else:
            msg.info(title='Pre-processing will happen on CPU')

        try:
            self.__preprocessing_nlp__ = spacy.load(preprocessing_nlp_model,
                                                    exclude=['attribute_ruler', 'lemmatizer', 'ner'])
        except:
            raise RuntimeError(
                f'There was an error while loading en_core_web_sm\n To rectify try running:\n pip install -U {models_url[preprocessing_nlp_model]}')

    def _clean_cache(self):
        ids = [sha256(text.encode('utf-8')).hexdigest() for text in self._input_text]
        for key in self._cache.keys():
            if str(key) not in ids:
                self._cache.pop(str(key))

    def __getitem__(self, item):
        if self._use_cache:
            self._clean_cache()
        to_process = self._input_text[item]
        if type(to_process) == str:
            to_process = [to_process]
        data = []
        if self._verbose:
            msg.info('Processing input data!!!')
        for text in tqdm(to_process, disable=not self._verbose):
            file_id = sha256(text.encode('utf-8')).hexdigest()
            if self._use_cache and self._cache.get(file_id) is not None:
                data.append(self._cache[file_id])
            else:
                preamble_doc, judgement_doc = split_main_judgement_to_preamble_and_judgement(text=text,
                                                                                             sentence_splitting_nlp=self.__preprocessing_nlp__,
                                                                                             mini_batch_size=self._mini_batch_size)
                _processed_data = {'file_id': file_id, "preamble_doc": preamble_doc, "judgement_doc": judgement_doc,
                                   "original_text": text}
                if self._use_cache:
                    self._cache[file_id] = _processed_data
                data.append(_processed_data)
        if len(data) == 1:
            return data[0]
        else:
            return data

    def __len__(self):
        return len(self._input_text)

    def __call__(self):
        if len(self._input_text) == 1:
            return self._input_text[0]
        else:
            return self._input_text

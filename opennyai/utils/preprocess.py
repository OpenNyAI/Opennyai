import spacy
from hashlib import sha256
from wasabi import msg
from tqdm import tqdm
from .download import PIP_INSTALLER_URLS, install
from .sentencizer import split_main_judgement_to_preamble_and_judgement


class Data:
    def __init__(self, input_text, preprocessing_nlp_model='en_core_web_trf', mini_batch_size=40000, use_gpu=True,
                 use_cache=True, verbose=False):
        self.__input_text__ = input_text
        self.__mini_batch_size__ = mini_batch_size
        self.__verbose__ = verbose
        self.__use_cache__ = use_cache
        if self.__use_cache__:
            self.__cache__ = {}
        if isinstance(self.__input_text__, str):
            self.__input_text__ = [self.__input_text__]
        elif isinstance(self.__input_text__, list) and len(self.__input_text__) >= 1:
            pass
        else:
            raise RuntimeError('No input or wrong given, we accept input as string or list of strings')
        if preprocessing_nlp_model not in spacy.util.get_installed_models():
            msg.info(f'Installing {preprocessing_nlp_model} this is a one time process!!')
            if PIP_INSTALLER_URLS.get(preprocessing_nlp_model) is not None and preprocessing_nlp_model in [
                'en_core_web_trf',
                'en_core_web_sm',
                'en_core_web_md']:
                install(PIP_INSTALLER_URLS[preprocessing_nlp_model])
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
                f'There was an error while loading en_core_web_sm\n To rectify try running:\n pip install -U {PIP_INSTALLER_URLS[preprocessing_nlp_model]}')

    def _clean_cache(self):
        ids = [sha256(text.encode('utf-8')).hexdigest() for text in self.__input_text__]
        for key in self.__cache__.keys():
            if str(key) not in ids:
                self.__cache__.pop(str(key))

    def __getitem__(self, item):
        if self.__use_cache__:
            self._clean_cache()
        to_process = self.__input_text__[item]
        if isinstance(to_process, str):
            to_process = [to_process]
        data = []
        if self.__verbose__:
            msg.info('Processing input data!!!')
        for text in tqdm(to_process, disable=not self.__verbose__):
            file_id = sha256(text.encode('utf-8')).hexdigest()
            if self.__use_cache__ and self.__cache__.get(file_id) is not None:
                data.append(self.__cache__[file_id])
            else:
                preamble_doc, judgement_doc = split_main_judgement_to_preamble_and_judgement(text=text,
                                                                                             sentence_splitting_nlp=self.__preprocessing_nlp__,
                                                                                             mini_batch_size=self.__mini_batch_size__)
                _processed_data = {'file_id': file_id, "preamble_doc": preamble_doc, "judgement_doc": judgement_doc,
                                   "original_text": text}
                if self.__use_cache__:
                    self.__cache__[file_id] = _processed_data
                data.append(_processed_data)
        if len(data) == 1:
            return data[0]
        else:
            return data

    def __len__(self):
        return len(self.__input_text__)

    def __str__(self):
        if len(self.__input_text__) == 1:
            return self.__input_text__[0]
        else:
            return str(self.__input_text__)

    def __contains__(self, item):
        return True if item in self.__input_text__ else False

    def __add__(self, other):
        if isinstance(other, str):
            self.__input_text__.append(other)
        else:
            TypeError('Only str object can be added')

    def append(self, other):
        self.__add__(other)

    def pop(self, index):
        if isinstance(index, int):
            self.__input_text__.pop(index)
        else:
            TypeError("'str' object cannot be interpreted as an integer")

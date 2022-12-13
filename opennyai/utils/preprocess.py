from hashlib import sha256
from typing import Union

import spacy
from tqdm import tqdm
from wasabi import msg

from .download import PIP_INSTALLER_URLS, install
from .sentencizer import split_main_judgement_to_preamble_and_judgement


class Data:
    def __init__(self, input_text: Union[str, list], preprocessing_nlp_model: str = 'en_core_web_trf',
                 mini_batch_size: int = 40000, use_gpu: bool = True,
                 use_cache: bool = True, verbose: bool = False, file_ids: list = []):
        """Returns object of Data class.
         It is used for common preprocessing of all the components present in this library.
        Args:
            input_text (string or list): This is where you will provide input as string or list of strings
            preprocessing_nlp_model (string): Accepts a model name of spacy as string that will be used for processing
            available models are 'en_core_web_trf', 'en_core_web_sm', 'en_core_web_md'
            mini_batch_size (int): This accepts an int as batch size for processing of a document,
             if length of document is bigger that given batch size it will be chunked and then processed.
            use_gpu (bool): Functionality to give a choice whether to use GPU for processing or not
             Setting it True doesn't ensure GPU will be utilized it need proper support libraries as mentioned in
             documentation
            use_cache (bool): Set it to true if you want to enable caching while preprocessing
            verbose (bool): Set it to if you want to see progress bar while processing happens
            file_ids (list): list of custom file ids to use with documents

            Examples::
            >>> text = 'Section 319 Cr.P.C. contemplates a situation where the evidence adduced by the prosecution for Respondent No.3-G. Sambiah on 20th June 1984'
            >>> data1 = Data(text)
            >>> data2 = Data([text,text])
            >>> data1.append(text)
            >>> data2.pop(1)
            >>> data1+text
            >>> len(data1)
            >>> text in data1
            >>> processed_data1 = [i for i in data1]
            >>> processed_data2 = data2[0]
        """
        self.__input_text__ = input_text
        self.__mini_batch_size__ = mini_batch_size
        self.__verbose__ = verbose
        self.__use_cache__ = use_cache
        self.__file_ids__ = file_ids
        if self.__use_cache__:
            self.__cache__ = {}
        if self.__file_ids__ and (len(self.__file_ids__) != len(self.__input_text__)):
            raise RuntimeError('Count of file_ids not equal to count of input text')
        if isinstance(self.__input_text__, str):
            self.__input_text__ = [self.__input_text__]
        if self.__file_ids__:
            texts = []
            ids = []
            for index, text in enumerate(self.__input_text__):
                if text.strip():
                    texts.append(text)
                    ids.append(self.__file_ids__[index])
            self.__input_text__ = texts
            self.__file_ids__ = ids
        else:
            self.__input_text__ = [text for text in self.__input_text__ if text.strip()]
        if isinstance(self.__input_text__, list) and len(self.__input_text__) >= 1 and all(
                isinstance(item, str) for item in self.__input_text__):
            pass
        else:
            raise RuntimeError('No input or wrong given, we accept input as string or list of strings')
        if preprocessing_nlp_model not in spacy.util.get_installed_models():
            msg.info(f'Installing {preprocessing_nlp_model}. This is a one time process!!')
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
                    msg.info(title='Pre-processing will happen on GPU!')
                else:
                    msg.info(title='Pre-processing will happen on CPU!')
                spacy.prefer_gpu()
            except:
                msg.info(title='Pre-processing will happen on CPU!')
        else:
            msg.info(title='Pre-processing will happen on CPU!')

        try:
            self.__preprocessing_nlp__ = spacy.load(preprocessing_nlp_model,
                                                    exclude=['lemmatizer', 'ner'])
        except:
            raise RuntimeError(
                f'There was an error while loading {preprocessing_nlp_model}\n To rectify try running:\n pip install -U {PIP_INSTALLER_URLS[preprocessing_nlp_model]}')

    def _clean_cache(self):
        if not self.__file_ids__:
            ids = [sha256(text.encode('utf-8')).hexdigest() for text in self.__input_text__]
        else:
            ids = self.__file_ids__
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
        for index, text in tqdm(enumerate(to_process), disable=not self.__verbose__, total=len(to_process)):
            if self.__file_ids__:
                file_id = self.__file_ids__[item]
            else:
                file_id = sha256(text.encode('utf-8')).hexdigest()
            if self.__use_cache__ and self.__cache__.get(file_id) is not None:
                data.append(self.__cache__[file_id])
            else:
                original_text = text
                text = text.encode(encoding='ascii', errors='ignore').decode()
                preamble_doc, judgement_doc = split_main_judgement_to_preamble_and_judgement(text=text,
                                                                                             sentence_splitting_nlp=self.__preprocessing_nlp__,
                                                                                             mini_batch_size=self.__mini_batch_size__)
                preamble_doc = spacy.tokens.Doc.from_docs([i.as_doc() for i in preamble_doc.sents])
                judgement_doc = spacy.tokens.Doc.from_docs([i.as_doc() for i in judgement_doc.sents])
                if preamble_doc is None:
                    preamble_doc = self.__preprocessing_nlp__('')
                if judgement_doc is None:
                    judgement_doc = self.__preprocessing_nlp__('')
                _processed_data = {'file_id': file_id,
                                   "preamble_doc": preamble_doc,
                                   "judgement_doc": judgement_doc,
                                   "original_text": original_text}
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

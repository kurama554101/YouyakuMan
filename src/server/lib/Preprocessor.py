from pytorch_pretrained_bert import BertTokenizer

from lib.LangFactory import LangFactory
import pdb
from enum import Enum


class InferInputType(Enum):
    INPUT_TXT_FILE = 0
    INPUT_RAW_TXT  = 1


class Preprocessor:
    def __init__(self, config, pretrained_tokenizer_type='bert-base-uncased', input_type=InferInputType.INPUT_RAW_TXT, super_long=True, lang="jp", translator=None):
        self._input_type = input_type
        self._super_long = super_long
        self._translator = translator
        self._langfac    = LangFactory(lang=lang, config=config)
        self._tokenizer  = BertTokenizer.from_pretrained(pretrained_tokenizer_type, do_lower_case=True)
    
    def __call__(self, input_data):
        self._data = []
        self._load_data(input_data=input_data)
        # If rawdata isnt modelized, use google translation to translate to English
        if self._langfac.stat is 'Invalid':
            self._translate()
        
        # Outsource suitable line splitter
        self._texts = self._langfac.toolkit.linesplit(self._rawtexts)
        # Outsource suitable tokenizer
        self._token, self._token_id = self._langfac.toolkit.tokenizer(self._texts)
        self._generate_results()
        return self._data

    def _generate_results(self):
        if not self._super_long:
            _, _ = self._add_result(self._fname, self._token_id)
        else:
            # Initialize indexes for while loop
            src_start, token_start, src_end = 0, 0, 1
            while src_end != 0:
                token_end, src_end = self._add_result(self._fname, self._token_id,
                                                      src_start, token_start)
                token_start = token_end
                src_start = src_end

    def _add_result(self, fname, token_all, src_start=0, token_start=0):
        results, (token_end, src_end) = self._all_tofixlen(token_all, src_start, token_start)
        token, clss, segs, labels, mask, mask_cls, src = results
        self._data.append({'fname': fname,
                          'src': token,
                          'labels': labels,
                          'segs': segs,
                          'mask': mask,
                          'mask_cls': mask_cls,
                          'clss': clss,
                          'src_str': src})
        return token_end, src_end

    def _load_data(self, input_data):
        if self._input_type == InferInputType.INPUT_TXT_FILE:
            path = input_data
            self._fname = path.split('/')[-1].split('.')[0]
            with open(path, 'r', encoding='utf-8_sig', errors='ignore') as f:
                self._rawtexts = f.readlines()
            self._rawtexts = ' '.join(self._rawtexts)
        elif self._input_type == InferInputType.INPUT_RAW_TXT:
            self._fname = ""  # TODO : need to input ?
            self._rawtexts = input_data
        else:
            raise NotImplementedError("{} input type is not supported!".format(self._input_type))

    def _translate(self):
        texts = self._rawtexts
        self._texts = self._translator.input(texts)

    def _list_tokenize(self):
        src_tokenize = []
        for src in self._texts:
            src_subtokens = self._tokenizer.tokenize(src)
            src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
            src_subtoken_idxs = self._tokenizer.convert_tokens_to_ids(src_subtokens)
            src_tokenize += src_subtoken_idxs
        return src_tokenize

    def _all_tofixlen(self, token, src_start, token_start):
        # Tune All shit into 512 length
        token_end = 0
        src_end = 0
        token = token[token_start:]
        src = self._texts[src_start:]
        clss = [i for i, x in enumerate(token) if x == self._langfac.toolkit.cls_id]
        if len(token) > 512:
            clss, token, token_stop, src, src_stop = self._length512(src, token, clss)
            token_end = token_start + token_stop
            src_end = src_start + src_stop
        labels = [0] * len(clss)
        mask = ([True] * len(token)) + ([False] * (512 - len(token)))
        mask_cls = [True] * len(clss)
        token = token + ([self._langfac.toolkit.mask_id] * (512 - len(token)))
        segs = []
        flag = 1
        for idx in token:
            if idx == self._langfac.toolkit.cls_id:
                flag = not flag
            segs.append(int(flag))
        return (token, clss, segs, labels, mask, mask_cls, src), (token_end, src_end)

    @staticmethod
    def _length512(src, token, clss):
        if max(clss) > 512:
            src_stop = [x > 512 for x in clss].index(True) - 1
        else:
            src_stop = len(clss) - 1
        token_stop = clss[src_stop]
        clss = clss[:src_stop]
        src = src[:src_stop]
        token = token[:token_stop]
        return clss, token, token_stop, src, src_stop

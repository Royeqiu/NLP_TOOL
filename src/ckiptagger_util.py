from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import os
from NLP_Tool_Constant import CKIPTAGGER_MODEL_PATH


class Ckiptagger_Util():
    def __init__(self, sentence_segmentation=False,use_cuda=False):
        model_path = os.path.join(CKIPTAGGER_MODEL_PATH, 'data')
        if not os.path.exists(model_path):
            data_utils.download_data_gdown(CKIPTAGGER_MODEL_PATH)
        if use_cuda:
            disable_cuda=False
        self.ws = WS(model_path,disable_cuda=disable_cuda)
        self.pos = POS(model_path,disable_cuda=disable_cuda)
        self.ner = NER(model_path,disable_cuda=disable_cuda)

    def pipe(self, texts, use_delim=False,delim_set={",", "。", ":", "?", "!", ";"}):
        word_sentence_list = self.get_word_seg(texts, sentence_segmentation=use_delim,
                                               segment_delimiter_set=delim_set)
        pos_sentence_list = self.get_pos_tag(word_sentence_list)

        entity_sentence_list = self.get_ner_tag(word_sentence_list, pos_sentence_list)
        return (word_sentence_list, pos_sentence_list, entity_sentence_list)

    def get_word_seg(self, texts, use_delim=False, delim_set={",", "。", ":", "?", "!", ";"}, recommend_dictionary={},
                     coerce_dictionary={}):
        word_sentence_list = self.ws(texts, sentence_segmentation=use_delim, segment_delimiter_set=delim_set,
                                     recommend_dictionary=recommend_dictionary, coerce_dictionary=coerce_dictionary)
        return word_sentence_list

    def get_pos_tag(self, word_sentence_list):
        pos_sentence_list = self.pos(word_sentence_list)
        return pos_sentence_list

    def get_ner_tag(self, word_sentence_list, pos_sentence_list):
        ner_sentence_list = self.ner(word_sentence_list, pos_sentence_list)
        return ner_sentence_list

    def set_up_dictionary(self,word_to_weight_dict):
        '''
        word_to_weight = {
            "土地公": 1,
            "土地婆": 1,
            "公有": 2,
            "": 1,
            "來亂的": "啦",
            "緯來體育台": 1,
        }
        '''
        dictionary = construct_dictionary(word_to_weight_dict)
        return dictionary
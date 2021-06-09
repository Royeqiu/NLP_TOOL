from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import os
from NLP_Tool_Constant import CKIPTAGGER_MODEL_PATH
class Ckiptagger_Util():
    def __init__(self,sentence_segmentation=False,use_cuda = False):
        if use_cuda:
            self.device = 0
        else:
            self.device = -1
        self.ws = CkipWordSegmenter(level=3,device=self.device)
        self.pos = CkipPosTagger(level=3,device=self.device)
        self.ner = CkipNerChunker(level=3,device=self.device)

    def pipe(self, texts, use_delim=False, delim_set='，,。：:；;！!？?'):
        word_sentence_list = self.get_word_seg(texts, use_delim=use_delim,
                                               delim_set=delim_set)
        pos_sentence_list = self.get_pos_tag(word_sentence_list)

        entity_sentence_list = self.get_ner_tag(word_sentence_list, pos_sentence_list)
        return (word_sentence_list,pos_sentence_list,entity_sentence_list)

    def get_word_seg(self, texts, use_delim=False,delim_set='，,。：:；;！!？?'):
        word_sentence_list = self.ws(texts,  use_delim=use_delim,delim_set=delim_set)
        return word_sentence_list

    def get_pos_tag(self,word_sentence_list):
        pos_sentence_list = self.pos(word_sentence_list)
        return pos_sentence_list

    def get_ner_tag(self,word_sentence_list,pos_sentence_list):
        ner_sentence_list = self.ner(word_sentence_list,pos_sentence_list)
        return ner_sentence_list


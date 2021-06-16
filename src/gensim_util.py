from gensim.models.phrases import Phrases
import os
import json
from NLP_Tool_Constant import GENSIM_CONST

class Phrase_Model():

    def __init__(self,model_path=None,model_name='phrase_model'):
        self.hierarchical_model = {}
        self.config = None
        if model_path is not None:
            target_path = os.path.join(model_path,model_name)
            with open(os.path.join(target_path,'config.json'),'r',encoding='utf-8') as cfg:
                self.config = json.loads(cfg.read())
                cfg.close()
            self.gram_limit = self.config['gram_limit']
            self.connector_word = self.config['connector_word']
            for gram_count in range(1,self.gram_limit):
                gram_flag = GENSIM_CONST.GRAM_TYPE_FLAG_DICT[gram_count]
                self.hierarchical_model[gram_flag] = Phrases.load(os.path.join(target_path,'{}.mod'.format(gram_flag)))

    def train_model(self,sentences,min_count = 5, threshold = 10, gram_limit = 3):
        self.connector_word = ['之','中','的','且']

        self.gram_limit = gram_limit
        gram_sentences = dict()
        gram_sentences[0] = sentences
        for gram_count in range(1,gram_limit):
            target_sentences = gram_sentences[gram_count-1]
            GRAM_FLAG = GENSIM_CONST.GRAM_TYPE_FLAG_DICT[gram_count]
            phrase_model = Phrases(target_sentences, min_count=min_count, threshold=threshold,connector_words=set(self.connector_word))
            self.hierarchical_model[GRAM_FLAG] = phrase_model
            gram_sentences[gram_count] = [sentence for sentence in phrase_model[target_sentences]]

    def get_phrase(self,sentences,gram_flag):
        return [phrase_sentence for phrase_sentence in self.hierarchical_model[gram_flag][sentences]]

    def save(self,model_path,model_name='phrase_model'):
        for gram_count in range(1,self.gram_limit):
            gram_flag = GENSIM_CONST.GRAM_TYPE_FLAG_DICT[gram_count]
            self.hierarchical_model[gram_flag].save(os.path.join(model_path,model_name,'{}.mod'.format(gram_flag)))
        if self.config is None:
            self.config = dict()
            self.config['gram_limit'] = self.gram_limit
            self.config['connector_word'] = self.connector_word
        with open(os.path.join(model_path,model_name,'config.json'),'w',encoding='utf-8') as cfg:
            cfg.write(json.dumps(self.config))
            cfg.close()

class Gensim_Util():

    def __init__(self,model_path=None,model_name='phrase_model'):
        self.phrase_model = Phrase_Model(model_path,model_name)

    def build_phrase_model(self,sentences,min_count = 5, threshold = 10, gram_limit = 3):
        self.phrase_model.train_model(sentences,min_count=min_count,threshold=threshold,gram_limit=gram_limit)

    def inference(self, sentences, gram_limit=3):
        gram_sentences = dict()
        gram_sentences[0] = sentences
        for gram_count in range(1,gram_limit):
            gram_flag = GENSIM_CONST.GRAM_TYPE_FLAG_DICT[gram_count]
            gram_sentences[gram_count]= self.phrase_model.get_phrase(gram_sentences[gram_count-1],gram_flag)

        return gram_sentences

    def save_model(self,model_path,model_name='phrase_model'):
        target_path = os.path.join(model_path,model_name)
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        self.phrase_model.save(model_path,model_name)


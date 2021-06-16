from gensim.models.phrases import Phrases
import json

class Gensim_Util():

    def __init__(self):
        pass

    def build_phrase_model(self,sentences,min_count = 5, threshold = 10 ):
        self.connector_word = {'之','中','的','且'}
        self.phrase_model = Phrases(sentences, min_count=min_count, threshold=threshold,connector_words=self.connector_word)

    def get_phrases(self,sentences):
        return self.phrase_model[sentences]

    def save_model(self,path):
        self.phrase_model.save(path)



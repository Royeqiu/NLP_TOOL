#from gensim.test.utils import common_texts
from src.ckiptagger_util import Ckiptagger_Util
from NLP_Tool import NLP_Tool
from gensim.models.phrases import Phrases
import pickle
import json
nlp_tool = NLP_Tool(load_spacy_model=False,load_jieba=True)
print('load_ckiptegger')
wiki_path = './example/Embeddings/wiki_texts.txt'
write_path = './example/Embeddings/wiki_phrase_texts.txt'
ckiptagger = Ckiptagger_Util(use_cuda=True)
print('load_wiki_article')
file = open(wiki_path,'r',encoding='utf-8')
tokens = []
batch_size = 200
batch = []
with open(write_path,'w',encoding='utf-8') as output:
    for index,data in enumerate(file):
        if index % batch_size == 0 and len(batch)!=0:
            results = ckiptagger.get_word_seg(batch)
            print(results)
            for result in results:
                t = [token for token in result if token != ' ']
                output.write(json.dumps(t,ensure_ascii=False)+ '\n')
                tokens.append(t)
            batch = []

        if index % 100 == 0:
            print(index)
        batch.append(data.strip())

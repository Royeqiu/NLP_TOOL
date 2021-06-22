#load phrase_model to accelerate the model train.
from gensim.models import FastText
from src.gensim_util import Gensim_Util
from NLP_Tool_Constant import GENSIM_CONST
import os
phrase_model_path = './model'
phrase_model_name = 'phrase_model'
example_file_path = './example_data/wiki_phrase_texts.txt'
w2v_model_path = './model'
w2v_model_name = 'FastText_all'
gensim_util = Gensim_Util(task_type=GENSIM_CONST.PHRASE_TASK, model_path= phrase_model_path, model_name=phrase_model_name)
wiki_articles = gensim_util.load_wiki_articles(example_file_path,articles_count=500000)
print(gensim_util.phrase_model.gram_limit)
wiki_phrase_articles=gensim_util.inference(wiki_articles)
print('Gram_Count: ',wiki_phrase_articles.keys())
#min_count = 5 make sure the typo doesn't appear in our dictionary
model = FastText(wiki_phrase_articles[2]+wiki_phrase_articles[1]+wiki_phrase_articles[0], min_count=5)
model.save(os.path.join(w2v_model_path,w2v_model_name,'w2v.mod'))
print(model.wv['恐慌'])
print(model.wv.similar_by_word('恐慌',50))
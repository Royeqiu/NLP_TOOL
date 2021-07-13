import os
from src.ckiptagger_util import Ckiptagger_Util
from src.gensim_util import Gensim_Util
from NLP_Tool_Constant import GENSIM_CONST
from NLP_Tool import NLP_Tool
from gensim.models import FastText
w2v_model_path = './model/FastText_all'
w2v_model_name = 'w2v.mod'
phrase_model_path = './model'
phrase_model_name = 'phrase_model'
gensim_util = Gensim_Util(task_type=GENSIM_CONST.PHRASE_TASK, model_path= phrase_model_path, model_name=phrase_model_name)
nlp_tool = NLP_Tool(load_spacy_model=False)
gensim_util_w2v = Gensim_Util(task_type=GENSIM_CONST.W2V_TASK,model_path= w2v_model_path,model_name= w2v_model_name)
#gensim_util_w2v
#model=FastText.load(os.path.join(w2v_model_path,w2v_model_name))
ckiptagger = Ckiptagger_Util()
word_seg = ckiptagger.get_word_seg(['動態隨機一般均衡模型'])
phrases = gensim_util.inference(word_seg[0])
print(word_seg)
print(phrases[0],phrases[1],phrases[2])
# get vecs by phrases
words = [word for word in phrases[2] if gensim_util_w2v.w2v_model.contains(word)]
vecs = gensim_util_w2v.inference(words)
#print(vecs)
avg_vec = nlp_tool.get_avg_vector(vecs,weights=[2,2,-1],vector_size=100)
print(gensim_util_w2v.w2v_model.get_similar_phrase_by_vec(avg_vec,50))

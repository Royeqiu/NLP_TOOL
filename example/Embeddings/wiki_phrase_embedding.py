from gensim.models import Word2Vec
from src.gensim_util import Gensim_Util
from NLP_Tool_Constant import GENSIM_CONST
model_path = './model'
model_name = 'phrase_model'
gensim_util = Gensim_Util(task_type=GENSIM_CONST.PHRASE_TASK,model_path= model_path,model_name=model_name)

from src.gensim_util import Gensim_Util
import json
from NLP_Tool_Constant import GENSIM_CONST

def main():
    example_file_path = './example_data/wiki_phrase_texts.txt'
    example_model_path = './model'
    model_name = 'phrase_model'
    task_type = GENSIM_CONST.PHRASE_TASK
    #load wiki-articles count
    articles_count = 500000
    gensim_util = Gensim_Util(task_type = GENSIM_CONST.PHRASE_TASK)
    docs = gensim_util.load_wiki_articles(example_file_path,articles_count=articles_count)

    article_counts = len(docs)
    gensim_util.build_phrase_model(sentences=docs , min_count=10,threshold=10,gram_limit=3)
    gensim_util.save_model(example_model_path.format(article_counts),model_name = model_name)
    gram_sentences = gensim_util.inference(docs[0:3])
    for key in gram_sentences.keys():
        print(key)
        print(gram_sentences[key])
        print()

    # init by loading phrase_model
    gensim_util = Gensim_Util(task_type=task_type,model_path=example_model_path,model_name=model_name)
    gram_sentences = gensim_util.inference(docs[0:3])
    for key in gram_sentences.keys():
        print(key)
        print(gram_sentences[key])
        print()
if __name__ == '__main__':
    main()
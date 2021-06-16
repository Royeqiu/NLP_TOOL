from src.gensim_util import Gensim_Util
import json

def main():
    example_file_path = './example_data/wiki_phrase_texts.txt'
    example_model_path = './model'
    articles_count = 10000
    gensim_util = Gensim_Util()
    file = open(example_file_path,'r',encoding='utf-8')
    docs = []
    for index,data in enumerate(file):
        if index==articles_count:
            break
        if index % 1000 == 0:
            print(index)
        sentences = json.loads(data.strip())
        docs.append(sentences)

    article_counts = len(docs)
    gensim_util.build_phrase_model(sentences=docs , min_count=10,threshold=10,gram_limit=3)
    gensim_util.save_model(example_model_path.format(article_counts))
    gram_sentences = gensim_util.inference(docs[0:3])
    for key in gram_sentences.keys():
        print(key)
        print(gram_sentences[key])
        print()

    # init by loading phrase_model
    gensim_util = Gensim_Util(model_path='./model',model_name='phrase_model')
    gram_sentences = gensim_util.inference(docs[0:3])
    for key in gram_sentences.keys():
        print(key)
        print(gram_sentences[key])
        print()
if __name__ == '__main__':
    main()
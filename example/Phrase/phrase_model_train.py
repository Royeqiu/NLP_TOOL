from src.gensim_util import Gensim_Util

def main():
    example_file_path = './example_data/wiki_phrase_texts.txt'
    example_model_path = './example/Phrase/wiki_phrase_model_{}_bi.mod'
    articles_count = 500000
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
    gensim_util.build_phrase_model(sentences=docs , min_count=10,threshold=10)
    gensim_util.save_model(example_model_path.format(article_counts))
    for res in gensim_util.get_phrases(docs[0:10]):
        print(res)

if __name__ == '__main__':
    main()
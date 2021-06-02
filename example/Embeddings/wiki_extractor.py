import logging
#from NLP_Tool import NLP_Tool
import opencc
from gensim.corpora import WikiCorpus
#can be downloaded from https://dumps.wikimedia.org/zhwiki/
output_path = './example/Embeddings/wiki_texts.txt'
wiki_corpus_path = './example_data/zhwiki-20210520-pages-articles-multistream.xml.bz2'
converter = opencc.OpenCC('s2tw.json')
if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    wiki_corpus = WikiCorpus(wiki_corpus_path, dictionary={})
    texts_num = 0
    with open(output_path,'w',encoding='utf-8') as output:
        for index,text in enumerate(wiki_corpus.get_texts()):
            if index % 1000==0:
                print(index)
            concated_str = converter.convert(' '.join(text) + '\n')
            output.write(concated_str)

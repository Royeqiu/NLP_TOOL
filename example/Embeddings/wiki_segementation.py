import pickle
import json
from src.ckiptagger_util import Ckiptagger_Util

print('load_ckiptegger')
wiki_path = './example/Embeddings/wiki_texts.txt'
write_path = './example_data/wiki_phrase_texts.txt'
ckiptagger = Ckiptagger_Util(use_cuda=True)
print('load_wiki_article')
file = open(wiki_path,'r',encoding='utf-8')
tokens = []
batch_size = 200
batch = []
with open(write_path,'w',encoding='utf-8') as output:
    for index,data in enumerate(file):
        if index % batch_size == 0 and len(batch)!=0:
            print(index)
            results = ckiptagger.get_word_seg(batch)
            for result in results:
                t = [token for token in result if token != ' ']
                output.write(json.dumps(t,ensure_ascii=False)+ '\n')
                tokens.append(t)
            batch = []
        batch.append(data.strip())
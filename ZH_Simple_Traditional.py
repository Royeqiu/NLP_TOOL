import json
import jieba

from hanziconv import HanziConv

def load_jieba_dic(dic_path):
    jieba.load_userdict(dic_path)

def load_dic(dic_path):
    json_str = ''
    with open(dic_path,'r',encoding='utf-8') as op:
        for text in op:
            json_str += text.strip()
    dictionary=json.loads(json_str)
    simple_to_traditional_dict = dict()
    traditional_to_simple_dict = dict()
    for index in range(0,len(dictionary['dic']),2):
        simple_to_traditional_dict[dictionary['dic'][index]] = dictionary['dic'][index+1]
        traditional_to_simple_dict[dictionary['dic'][index+1]] = dictionary['dic'][index]
    return simple_to_traditional_dict,traditional_to_simple_dict


def to_simple(sentence):
    trans_sentence = ''
    for word in jieba.cut(sentence):
        print(word)

        if word in traditiona_to_simple_dict.keys():
            trans_sentence += traditiona_to_simple_dict[word]
        else:
            trans_sentence += word
    return trans_sentence

def to_traditional(sentence):
    trans_sentence = ''
    for word in jieba.cut(sentence):
        if word in simple_to_traditional_dict.keys():
            trans_sentence += simple_to_traditional_dict[word]
        else:
            trans_sentence += word
    return trans_sentence

simple_to_traditional_dict, traditiona_to_simple_dict = load_dic('NLP_TOOL/tr_sim_dic.txt')
load_jieba_dic('NLP_TOOL/Taiwan_Jieba_dic.txt')
if __name__ == '__main__':
    sen = '我很喜歡吃飯糰'
    print(to_simple(sen))
    print(HanziConv.toSimplified(sen))
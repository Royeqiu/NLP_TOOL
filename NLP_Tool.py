import spacy
import re
import numpy as np
import math
from spacy.tokens import Doc
from pyhanlp import *
import os
import NLP_Tool_Constant as CONST
from src.ckiptagger_util import Ckiptagger_Util
from src.bert_util import Bert_Util


class NLP_Tool:

    def __init__(self, load_lg_corpus=True, load_spacy_model=True, load_hanlp_model=False, enable_hanlp_ner=False,
                 load_jieba=False, load_ckiptagger=False, load_bert_model=False):
        if load_spacy_model:
            self.nlp = self.load_spacy(load_lg_corpus)
            self.load_stop_word()
        if load_hanlp_model:
            self.load_ch_stop_word()
            self.hanlp = self.load_hanlp(enable_hanlp_ner)
        if load_jieba:
            self.load_ch_stop_word()
            self.jieba = self.load_jieba()
        if load_jieba or load_hanlp_model:
            self.load_ch_stop_word()
        if load_ckiptagger:
            self.ckiptagger = self.load_ckiptagger()
        if load_bert_model:
            self.bert = self.load_bert()
        else:
            self.bert = None

    def get_bert_vec(self, sentences):
        if self.bert is None:
            self.bert = self.load_bert()
        return self.bert(sentences)

    def get_tokens_bert_vec(self, tokens, sentence, vec=None):
        cls_leng = 1
        start_index, end_index = self.get_index(term=tokens, sentence=sentence, skip_space=False)
        if vec is None:
            vec = self.get_bert_vec(sentence)[0]
        print(vec.shape)
        return vec[0][start_index + cls_leng:end_index + cls_leng]

    def load_bert(self):
        bert = Bert_Util()
        return bert

    def load_hanlp(self, enable_ner=False):
        hanlp = HanLP.newSegment().enableNameRecognize(enable_ner)

        return hanlp
    def load_ckiptagger(self):
        self.ckiptagger = Ckiptagger_Util()

    def load_spacy(self, load_lg_corpus=True):
        if load_lg_corpus:
            nlp = spacy.load('en_core_web_lg')
        else:
            nlp = spacy.load('en_core_web_sm')
        return nlp

    def load_jieba(self):
        import jieba

        return jieba

    def cal_idf(self, corpus):
        total_count = len(corpus)
        word_set = set()
        word_count_dict = dict()
        for sentence in corpus:
            for word in sentence:
                word_set.add(word)

        for word in word_set:
            word_count_dict[word] = 0
            for sentence in corpus:
                if word in sentence:
                    word_count_dict[word] += 1
            word_count_dict[word] = math.log(total_count / word_count_dict[word])
        return word_count_dict

    def load_stop_word(self):
        for word in self.nlp.Defaults.stop_words:
            lex = self.nlp.vocab[word]
            lex.is_stop = True

    def load_ch_stop_word(self):
        self.ch_stop_word_list = []
        stop_word_file = open(str(os.path.dirname(__file__)) + '/traditional_chinese_stop_word.txt', 'r',
                              encoding='utf-8')
        for term in stop_word_file:
            self.ch_stop_word_list.append(term.strip())
        stop_word_file.close()

    def tokenize_sentence(self, sentence):
        return [word.strip() for word in re.split('(\W+)?', sentence) if word.strip()]

    def remove_punctuation(self, text):
        mypunctuation = '!"#$&\'()*+-/:;<=>?@[\\]^_`{|}~*½。，.'
        regex = re.compile('[%s]' % re.escape(mypunctuation))
        text = regex.sub('', text)
        return text

    def get_tokens(self, text):
        doc = self.nlp(text)
        tokens = [token for token in doc if token]
        return tokens

    def get_sents(self, text):
        doc = self.nlp(text)
        return doc.sents

    def get_noun_chunks(self, text):
        doc = self.nlp(text)
        chunks = [chunk for chunk in doc.noun_chunks]
        return chunks

    def get_doc_similarity(self, text1, text2):
        doc = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc.similarity(doc2)

    def build_word_index(self, corpus, is_padding=False, is_unknown=False):
        word_set = set()
        word_to_index = dict()
        index_to_word = dict()
        if is_padding:
            word_set.add('PAD')
        if is_unknown:
            word_set.add('UNK')

        for word_list in corpus:
            for word in word_list:
                word_set.add(word)
        for word in word_set:
            word_to_index[word] = len(word_to_index)
            index_to_word[len(index_to_word)] = word
        return word_set, word_to_index, index_to_word

    def turn_to_one_hot(self, corpus, word_to_index):
        training_tokens_vec = []
        for tokens in corpus:
            vec = np.zeros(len(word_to_index))
            for token in tokens:
                if token in word_to_index.keys():
                    vec[word_to_index[token]] = 1
            training_tokens_vec.append(vec)
        return training_tokens_vec

    def get_cos_similarity(self, vec1, vec2):
        vec1_leng = 0
        for value in vec1:
            vec1_leng += (value * value)
        vec1_leng = math.sqrt(vec1_leng)
        vec2_leng = 0
        for value in vec2:
            vec2_leng += (value * value)
        vec2_leng = math.sqrt(vec2_leng)
        product = np.dot(vec1, vec2)

        return product / (vec1_leng * vec2_leng)

    def get_arc_similarity(self, vec1, vec2):
        return (1 - math.acos(self.get_cos_similarity(vec1, vec2)) / math.pi)

    def get_phrase_vector(self, text):
        avg_total_vector = np.zeros((300), dtype='f')
        tokens = self.get_tokens(text)
        num = 0
        for token in tokens:
            if token.has_vector:
                avg_total_vector += token.vector
                num += 1
            else:
                continue
        if num != 0:
            return avg_total_vector / num
        else:
            return None

    def is_stop_word(self, word):
        return self.nlp.vocab[word].is_stop

    def is_ch_stop_word(self, word):
        return word in self.ch_stop_word_list

    def transform_pos(self, pos):
        abbr_pos = ''
        if pos == 'VERB':
            abbr_pos = 'v'
        if pos == 'ADJ':
            abbr_pos = 'a'
        if pos == 'ADV':
            abbr_pos = 'adv'
        if pos == 'NOUN':
            abbr_pos = 'n'
        return abbr_pos

    def set_tokenizer(self, Tokenizer, vocab):
        self.nlp.tokenizer = Tokenizer(vocab)

    def get_avg_vector(self, vectors,vector_size=300):
        avg_vector = np.zeros((vector_size), dtype='f')
        for vector in vectors:
            avg_vector += vector
        if len(vectors) == 0:
            return avg_vector
        else:
            return avg_vector / len(vectors)

    def get_index(self, term, sentence, skip_space=True):

        base_index = sentence.index(term)
        skipped_space_count = 0
        if skip_space:
            pre_sentence = sentence[0:base_index]
            skipped_space_count = len(pre_sentence.split(' ')) - 1

        return base_index - skipped_space_count, base_index + len(term) - skipped_space_count

    def bert_embedding(self, sentences):
        return self.bert(sentences)

    def build_n_gram_corpus(self, corpus, n=3):
        n_gram_list = []
        for i in range(n):
            n_gram_list.append([])
        for sentence in corpus:
            n_tmp_list = []
            for i in range(n):
                n_tmp_list.append([])
            for term_index, term in enumerate(sentence):
                n_term = term
                for window_size in range(1, n):
                    pointer = term_index + window_size
                    if pointer > len(sentence) - 1:
                        break
                    n_term += sentence[pointer]
                    n_tmp_list[window_size - 1].append(n_term)
            for i in range(n):
                n_gram_list[i].append(n_tmp_list[i - 1])
        n_gram_list[0] = corpus
        return n_gram_list

    def get_index(self, term, sentence, skip_space=True):

        if term not in sentence:
            return -1, -1
        base_index = sentence.index(term)
        skipped_space_count = 0
        if skip_space:
            pre_sentence = sentence[0:base_index]
            skipped_space_count = len(pre_sentence.split(' ')) - 1

        return base_index - skipped_space_count, base_index + len(term) - skipped_space_count

    def bert_embedding(self, sentences):
        return self.bert(sentences)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

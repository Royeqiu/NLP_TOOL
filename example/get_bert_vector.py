from NLP_Tool import NLP_Tool

nlp_tool = NLP_Tool(load_lg_corpus=False,load_spacy_model=False,load_bert_model=True)

#token embedding in the sentence
token_vec = print(nlp_tool.get_tokens_bert_vec(term = '飯糰',sentence= '你喜歡吃飯糰嗎',vec=None))
print(token_vec)

# process multiple sentences
vecs = nlp_tool.get_bert_vec(['你喜歡吃飯糰嗎','我不喜歡','真的嗎?'])
print(vecs)
print(vecs[0].shape)
print(vecs[0])
print(vecs[1].shape)
print(vecs[1])
print(vecs.last_hidden_state.shape)
print(dir(vecs.last_hidden_state))

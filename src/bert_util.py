from transformers import BertModel,BertTokenizer


class Bert_Util():
    def __init__(self,tokenizer_path='bert-base-multilingual-uncased',model_path='bert-base-multilingual-uncased'):
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.bert_model = BertModel.from_pretrained(model_path)

    def __call__(self, text):
        return self.encode(text)

    def encode(self,text):

        inputs = self.bert_tokenizer(text, return_tensors="pt",padding=True)
        return self.bert_model(**inputs)
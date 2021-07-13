from sklearn.metrics import f1_score,accuracy_score
from nltk.translate import gleu_score,bleu_score,nist_score
from jiwer import wer
from sklearn.metrics import confusion_matrix


def split_sentence(labels,predictions,split_sign):
    split_labels = []
    split_predictions = []
    for label,prediction in zip(labels,predictions):
        chars = []
        for single_l in label:
            if split_sign is not None:
                chars.append(single_l.split(split_sign))
            else:
                chars.append([char for char in single_l])
        split_labels.append(chars)
        if split_sign is not None:
            split_predictions.append(prediction.split(split_sign))
        else:
            split_predictions.append([char for char in prediction])
    return split_labels,split_predictions

class Evaluation_Method():
    def get_args(self,args):
        pass
    def evaluate(self,labels,predictions,args = None):
        objects = self.get_args(args)
        pass

class Accuracy(Evaluation_Method):

    def evaluate(self,labels,predictions,args=None):
        return accuracy_score(labels,predictions)

class F_Score(Evaluation_Method):

    def get_args(self,args):
        obj = dict()
        average = 'binary'
        if args is not None:
            if 'average' in args.keys():
                average = args['average']
        obj['average'] = average
        return obj
    def evaluate(self,labels,predictions,args = None):
        obj = self.get_args(args)
        average = obj['average']
        return f1_score(labels,predictions,average=average)

class Nist(Evaluation_Method):

    def get_args(self,args):
        obj = dict()
        n_gram = 5
        split_sign = None
        if args is not None:
            if 'n_gram' in args.keys():
                n_gram = args['n_gram']
            if 'split_sign' in args.keys():
                split_sign = ' '
        obj['n_gram'] = n_gram
        obj['split_sign'] = split_sign
        return obj

    def evaluate(self,labels,predictions,args = None):
        args = self.get_args(args)
        split_sign = args['split_sign']
        n_gram = args['n_gram']
        split_labels,split_predictions = split_sentence(labels,predictions,split_sign)
        print(split_predictions)
        print(split_labels)
        return nist_score.corpus_nist(split_labels,split_predictions,n_gram)

class Bleu(Evaluation_Method):
    def get_args(self,args):
        #len of weights means n-gram if len(wights) == 4 means 4-grams
        obj = dict()
        weights=(0.25, 0.25, 0.25, 0.25)
        smoothing_function=None
        auto_reweigh=False
        split_sign = None
        if args is not None:
            if 'weights' in args.keys():
                weights = args['weights']
            if 'smoothing_funtion' in args.keys():
                smoothing_function = args['smoothing_function']
            if 'auto_reweigh' in args.keys():
                auto_reweigh = args['auto_reweigh']
            if 'split_sign' in args.keys():
                split_sign = args['split_sign']
        obj['weights'] = weights
        obj['smoothing_function'] = smoothing_function
        obj['auto_reweigh'] = auto_reweigh
        obj['split_sign'] = split_sign
        return obj

    def evaluate(self,labels,predictions,args = None):
        obj = self.get_args(args)
        weights = obj['weights']
        smoothing_function = obj['smoothing_function']
        auto_reweigh = obj['auto_reweigh']
        split_sign = obj['split_sign']
        split_labels,split_predictions = split_sentence(labels,predictions,split_sign)
        return bleu_score.corpus_bleu(split_labels,split_predictions,weights,smoothing_function,auto_reweigh)


class Gleu(Evaluation_Method):

    def get_args(self,args):
        obj = dict()
        min_len = 1
        max_len = 4
        split_sign = None
        if args is not None:
            if 'min_len' in args.keys():
                min_len = args['min_len']
            if 'max_len' in args.keys():
                max_len = args['max_len']
            if 'split_sign' in args.keys():
                split_sign = args['split_sign']
        obj['min_len'] = min_len
        obj['max_len'] = max_len
        obj['split_sign'] = split_sign
        return obj

    def evaluate(self,labels,predictions,args = None):
        obj = self.get_args(args)
        min_len = obj['min_len']
        max_len = obj['max_len']
        split_sign = obj['split_sign']
        split_labels,split_predictions = split_sentence(labels,predictions,split_sign)
        return gleu_score.corpus_gleu(split_labels,split_predictions,min_len=min_len,max_len=max_len)

class Wer(Evaluation_Method):
    def evaluate(self,labels,predictions,args = None):
        return wer(labels,predictions)

class Confusion_Matrix(Evaluation_Method):

    def evaluate(self,labels,predictions,args = None):
        return confusion_matrix(labels,predictions)
from NLP_Evaluation_Tool import NLP_Evaluation_Tool
from NLP_Tool_Constant import NLP_EVALUATION_CONST

def acc_example():
    metrics = [NLP_EVALUATION_CONST.ACCURACY]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    print()
    predictions = [0,0,1,1,0,1,0,1,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1]
    print(nlp_eval.evaluate(labels,predictions))
    predictions = [0,2,1,3,0,1,4,1,0,0,1,2,3,2,1,2,4,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1,2,3,2,2,2,4,0,1,1]
    print(labels,predictions)

def f_score_example():
    metrics = [NLP_EVALUATION_CONST.F_SCORE]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    predictions = [0,0,1,1,0,1,0,1,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1]
    print(nlp_eval.evaluate(labels,predictions))
    predictions = [0,2,1,3,0,1,4,1,0,0,1,2,3,2,1,2,4,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1,2,3,2,2,2,4,0,1,1]
    args = {'average':NLP_EVALUATION_CONST.MACRO_MODE}
    print(nlp_eval.evaluate(labels,predictions,args))

def confusion_metrics():
    metrics = [NLP_EVALUATION_CONST.CONFUSION_MATRIX]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    predictions = [0,0,1,1,0,1,0,1,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1]
    print(nlp_eval.evaluate(labels,predictions))
    predictions = [0,2,1,3,0,1,4,1,0,0,1,2,3,2,1,2,4,0,0,1]
    labels = [0,0,1,0,0,1,1,1,0,0,1,2,3,2,2,2,4,0,1,1]
    print(nlp_eval.evaluate(labels,predictions))

def nist_example():
    metrics = [NLP_EVALUATION_CONST.NIST]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    labels = [['我很喜歡吃飯糰'],['也喜歡喝牛奶','我也喜喝牛奶'],['I love to eat rice']]
    predictions = ['我很討厭吃飯糰','也喜喝牛奶','I love to eat rice']
    print(nlp_eval.evaluate(labels,predictions))
    labels = [['I would like to get some milk','I want to get a bottle of milk'],['please write down the notes'],['I love to eat rice']]
    predictions = ['I want to get some milk','please write down the note','I love to eat rice']
    args = {'split_sign':' '}
    print(nlp_eval.evaluate(labels,predictions,args))

def bleu_example():
    metrics = [NLP_EVALUATION_CONST.BLEU]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    labels = [['我很喜歡吃飯糰'],['也喜歡喝牛奶','我也喜喝牛奶'],['I love to eat rice']]
    predictions = ['我很討厭吃飯糰','也喜喝牛奶','I love to eat rice']
    print(nlp_eval.evaluate(labels,predictions))
    labels = [['I would like to get some milk','I want to get some milk','I want to get a bottle of milk'],['please write down the note'],['I love to eat rice']]
    predictions = ['I want to get some milk','please write down the note','I love to eat rice']
    args = {'split_sign':' '}
    print(nlp_eval.evaluate(labels,predictions,args))

def gleu_example():
    metrics = [NLP_EVALUATION_CONST.GLEU]
    nlp_eval = NLP_Evaluation_Tool(metrics)
    labels = [['我很喜歡吃飯糰'],['也喜歡喝牛奶','我也喜喝牛奶'],['I love to eat rice']]
    predictions = ['我很討厭吃飯糰','也喜喝牛奶','I love to eat rice']
    print(nlp_eval.evaluate(labels,predictions))
    labels = [['I would like to get some milk','I want to get some milk','I want to get a bottle of milk'],['please write down the note'],['I love to eat rice']]
    predictions = ['I want to get some milk','please write down the note','I love to eat rice']
    args = {'split_sign':' '}
    print(nlp_eval.evaluate(labels,predictions,args))

acc_example()
f_score_example()
confusion_metrics()
nist_example()
bleu_example()
gleu_example()
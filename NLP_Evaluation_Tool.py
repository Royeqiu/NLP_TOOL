from eval.evaluation import *
from NLP_Tool_Constant import NLP_EVALUATION_CONST
class NLP_Evaluation_Tool:

    def __init__(self,evaluation_matrix):
        self.evaluation_matrix = [matrix for matrix in evaluation_matrix]
        self.evaluation_tool = []
        for matrix in self.evaluation_matrix:
            if matrix == NLP_EVALUATION_CONST.ACCURACY:
                self.evaluation_tool.append(Accuracy())
            elif matrix == NLP_EVALUATION_CONST.F_SCORE:
                self.evaluation_tool.append(F_Score())
            elif matrix == NLP_EVALUATION_CONST.NIST:
                self.evaluation_tool.append(Nist())
            elif matrix == NLP_EVALUATION_CONST.BLEU:
                self.evaluation_tool.append(Bleu())
            elif matrix == NLP_EVALUATION_CONST.GLEU:
                self.evaluation_tool.append(Gleu())
            elif matrix == NLP_EVALUATION_CONST.WER:
                self.evaluation_tool.append(Wer())
            elif matrix == NLP_EVALUATION_CONST.CONFUSION_MATRIX:
                self.evaluation_tool.append(Confusion_Matrix())
        print(self.evaluation_matrix,self.evaluation_tool)
    def evaluate(self,labels,predictions,args=None):
        result_dict = dict()
        for matrix, tool in zip(self.evaluation_matrix,self.evaluation_tool):
            result_dict[matrix] = tool.evaluate(labels,predictions,args)
        return result_dict
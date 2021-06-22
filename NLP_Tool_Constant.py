class CKIP_CONST():
    CKIPTAGGER_MODEL_PATH: str = './model/ckiptagger/'

class HANLP_CONST():
    HANLP_PEOPLE_TAG: str = 'nr'

class GENSIM_CONST():
    GRAM_TYPE_FLAG_DICT = {0:'uni',1:'bi',2:'tri',3:'quad',4:'pent'}
    PHRASE_TASK = 'phrase'
    W2V_TASK = 'w2v'
    W2V_FT_MODEL = 'FT'
    W2V_SG_MODEL = 'SG'
    W2V_CB_MODEL = 'CB'
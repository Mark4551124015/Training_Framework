from typing import Any
from tools.metrics import *

# Template of Evaluator
class Evaluator(object):
    def __init__(self) -> None:
        self.evl = {}
        self.init_all()

    def init_all(self):
        self.log = {}
        for k in self.evl:
            self.log[k] = []

    def __call__(self, predict, target) -> Any:
        for k in self.evl:
            sc = self.evl[k](predict, target)
            self.log[k] = sc
    
    def eval_one(self, predict, target):
        result = {}
        for k in self.evl:
            result[k] = self.evl[k](predict, target)
        return result
    
    def clear(self):
        for k in self.log:
            self.log[k] = []

    def start(self):
        self.clear()
        
    def stop(self):
        scores = self.get_score()
        self.clear()
        return scores

    def get_score(self):
        result = {}
        for k in self.log:
            result[k] = np.mean(self.log[k])
        return result

    def save(self, TYPE):
        with open(os.path.join("results/", TYPE + "_Evaluator.csv"), 'w') as f:
            score = self.get_score()
            f.write("Metric")
            for i, k in enumerate(score):
                f.write(f",{k}")
            f.write('\n')
            f.write(TYPE)
            for i, k in enumerate(score):
                f.write(f",{score[k]}")
            

# Instnaces of Evaluator

class Eval_Seg(Evaluator):
    def __init__(self) -> None:
        self.evl = {}
        self.evl['AUC'] = calc_auc
        self.evl['ACC'] = calc_acc
        self.evl['SEN'] = calc_sen
        self.evl['FDR'] = calc_fdr
        self.evl['SPE'] = calc_spe
        self.evl['GMEAN'] = calc_gmean
        self.evl['KAPPA'] = calc_kappa
        self.evl['IOU'] = calc_iou
        self.evl['DICE'] = calc_dice
        self.init_all()


class Eval_Cls(Evaluator):
    def __init__(self) -> None:
        self.evl = {}
        self.evl['AUC'] = calc_auc
        self.evl['ACC'] = calc_acc
        self.evl['SEN'] = calc_sen
        self.evl['SPE'] = calc_spe
        self.evl['GMEAN'] = calc_gmean
        self.evl['KAPPA'] = calc_kappa
        self.init_all()



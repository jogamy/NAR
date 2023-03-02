import os
import json

from .util import load_data
from ...utils.util import json_to_pred

# 나중에

class Evaluator():
    def __init__(self, pred_path, path) -> None:
        path = "test"
        self.srcs, self.real_poss, self.real_ners = load_data(path, False)
        # self.srcs 대문자되는지 봐야 해 대문자 되야해
        self.pred_poss, self.pred_ners = json_to_pred(pred_path)
        self.make_output()
        
        
    def make_output(self):
        path = "/root/ner/ner_output.txt"

        with open(path, 'w') as f:
            for pred_ners, real_ners, srcs in zip(self.pred_ners, self.real_ners, self.srcs):
                for pred_ner, real_ner, src in zip(pred_ners[0], real_ners, srcs):
                    f.write(f"{src} {real_ner} {pred_ner}\n")
                f.write("\n")
            f.write("\n")
        


        path = "/root/ner/pos_output.txt"
        with open(path, 'w') as f:
            for pred_ners, real_ners, srcs in zip(self.pred_poss, self.real_poss, self.srcs):
                for pred_ner, real_ner, src in zip(pred_ners[0], real_ners, srcs):
                    f.write(f"{src} {real_ner} {pred_ner}\n")
                f.write("\n")
            f.write("\n")
            
    
    def EM(self):
        count = 0
        for i in range(len(self.real_ners)):
            if self.pred_ners[i][0] == self.real_ners[i]:
                count += 1
        print(f"NER em: {count / len(self.real_ners)}")

        count = 0

        for i in range(len(self.real_poss)):
            if self.pred_poss[i][0] == self.real_poss[i]:
                count += 1
        print(f"pos em: {count / len(self.real_ners)}")


def analyze(pred_path, path, enc_tok, dec1_tok, dec2_tok):
    eval = Evaluator(pred_path, path)
    eval.EM()
    
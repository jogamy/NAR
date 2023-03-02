import os
import json
from collections import Counter

import numpy as np
from numpy import average
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from .util import load_data

class Evaluator(object):
    def __init__(self, pred_path, path):
        path = os.path.join(path, "test.txt")
        self.srcs, self.real_slot, self.real_intent = load_data(path)

        self.real_intent = [intent[0] for intent in self.real_intent]

        ## 여기를 exampels/utils/util로
        with open(pred_path, 'r') as f:
            pred = json.load(f)
        self.pred_slot = []
        self.pred_intent = []
        self.pred_intent_voting = []
        self.slot_prob = []
        self.ints_prob = []
        for i in range(len(pred['seqs1'])):
            self.pred_slot.append(pred['seqs1'][f"{i}"]['pred'][0])
            self.pred_intent_voting.append(pred['seqs2'][f"{i}"]['pred'][0])

            cur_int = pred['seqs2'][f"{i}"]['pred'][0]
            cur_int = Counter(cur_int)
            cur_int = cur_int.most_common(1)[0][0]
            self.pred_intent.append(cur_int)

            self.slot_prob.append(pred['probs1'][f"{i}"]['pred'][0])
            self.ints_prob.append(pred['probs2'][f"{i}"]['pred'][0])

        #################################################
        self.int_accs = []
        self.slot_accs = []
        self.slot_f1s = []
        self.sem_accs = []
        for i in range(len(self.pred_intent)):
            int_acc = self.accuracy([self.pred_intent[i]], [self.real_intent[i]])
            slot_acc = self.accuracy([self.pred_slot[i]], [self.real_slot[i]])
            slot_f1, _, _ = self.f1_score([self.pred_slot[i]], [self.real_slot[i]])
            sem_acc = self.semantic_acc([self.pred_slot[i]], [self.real_slot[i]], [self.pred_intent[i]], [self.real_intent[i]])
            self.int_accs.append(int_acc)
            self.slot_accs.append(slot_acc)
            self.slot_f1s.append(slot_f1)
            self.sem_accs.append(sem_acc)
        print(f"int acc: {average(self.int_accs)}")
        print(f"slot acc: {average(self.slot_accs)}")
        print(f"slot f1: {average(self.slot_f1s)}" )
        print(f"overall accs: {average(self.sem_accs)}")

        assert 1==0
        #################################################

        # sem_acc = self.semantic_acc(self.pred_slot, self.real_slot, self.pred_intent, self.real_intent)
        # print(f"semantic acc :  {sem_acc}")    
        # int_acc = self.accuracy(self.pred_intent, self.real_intent)
        # print(f"int acc :  {int_acc}")
        # slt_acc = self.accuracy(self.pred_slot, self.real_slot)
        # print(f"slt_acc :  {slt_acc}")
        slt_f1, _, _ = self.f1_score(self.pred_slot, self.real_slot)
        # print(f"slt_f1 :  {slt_f1}")


    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):
            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """
        tp, fp, fn = 0.0, 0.0, 0.0
        real_seg_list = []
        pred_seg_list = []
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1
            real_seg_list.append(seg)

            tp_ = 0
            j = 0
            pred_seg = set()
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    pred_seg.add((cur, j, k - 1))
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1
            pred_seg_list.append(pred_seg)

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        
        return f1, real_seg_list, pred_seg_list

    """
    Max frequency prediction. 
    """
    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
    

def analyze(pred_path, path, enc_tok, dec1_tok, dec2_tok):
    evaluator = Evaluator(pred_path, path)
    indices = []
    err_slot_f1s = []
    err_slot_accs = []
    err_intent_accs = []
    err_overall_accs = []

    
    pred_intents = []
    real_intents = []

    for i in range(len(evaluator.slot_f1s)):
        # condition = evaluator.real_intent[i] == "atis_flight_no#atis_airline"
        # condition = "#" in evaluator.real_intent[i]
        condition = evaluator.pred_intent[i] != evaluator.real_intent[i]
        if condition:
            # print(evaluator.real_intent[i])
            indices.append(i)
            err_intent_accs.append(evaluator.int_accs[i])
            err_slot_accs.append(evaluator.slot_accs[i])
            err_slot_f1s.append(evaluator.slot_f1s[i])
            err_overall_accs.append(evaluator.sem_accs[i])
            print(f"{evaluator.pred_intent[i]} -> {evaluator.real_intent[i]}")
            pred_intents.append(evaluator.pred_intent[i])
            real_intents.append(evaluator.real_intent[i])
            print(f"{evaluator.int_accs[i]}  / {evaluator.slot_accs[i]} / {evaluator.slot_f1s[i]}  / {evaluator.sem_accs[i]}")
            print(" ".join(evaluator.srcs[i]))
            # enc_tok.encode(evaluator.srcs[i])
            print("============================")
    
    # print(f"intent acc: {average(err_intent_accs)}")
    # print(f"slot acc : {average(err_slot_accs)}")
    # print(f"slot f1 :  {average(err_slot_f1s)}")
    # print(f"overall: {average(err_overall_accs)}")
    
    print(indices)
    print(len(indices))
    assert 1==0

    # CM #########################################
    
    # create a list of unique intent labels
    intent_labels = list(set(pred_intents + real_intents))

    # create a matrix to store the count of each predicted/true intent label pair
    count_matrix = np.zeros((len(intent_labels), len(intent_labels)))

    # fill the count matrix with the occurrence count of each predicted/true intent label pair
    for pred, real in zip(pred_intents, real_intents):
        pred_index = intent_labels.index(pred)
        real_index = intent_labels.index(real)
        count_matrix[pred_index, real_index] += 1
        

    # create a heatmap of the count matrix
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(count_matrix, cmap='Blues')

    # set the x-axis and y-axis labels
    ax.set_xticks(np.arange(len(intent_labels)) + 0.5)
    # ax.set_xticklabels(intent_labels, rotation=90)
    ax.set_xticklabels(intent_labels)
    ax.set_yticks(np.arange(len(intent_labels)) + 0.5)
    ax.set_yticklabels(intent_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # add the count as text to each box of the heatmap
    for i in range(len(intent_labels)):
        for j in range(len(intent_labels)):
            text = ax.text(j + 0.5, i + 0.5, int(count_matrix[i, j]),
                        ha='center', va='center', color='w')

    # add a color bar to the heatmap
    cbar = plt.colorbar(heatmap)

    # display the plot
    plt.show()

    # set the title and show the plot
    ax.set_title('Intent Prediction Heatmap')
    fig.savefig('/root/NAR/snips_intent_heatmap.png', dpi=300, bbox_inches='tight')

    
        
    
    
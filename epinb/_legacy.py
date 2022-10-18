import pandas as pd
import time
import numpy as np
from tqdm.notebook import tqdm
import os
import scipy.stats

# from numba import jit

class Seq_Prior:
    def __init__(self):
        #self.aa_prior = {k: v for k, v in zip(("A","D","C","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"), 
        #          (0.07,0.06,0.03,0.06,0.04,0.07,0.03,0.04,0.07,0.08,0.02,0.04,0.05,0.04,0.04,0.08,0.06,0.07,0.01,0.03) 
        #          )}
        self.aa_prior = {"A": 0.0777, "C": 0.0157, "D": 0.0530, "E": 0.0656, "F": 0.0405,
                         "G": 0.0691, "H": 0.0227, "I": 0.0591, "K": 0.0595, "L": 0.0960,
                         "M": 0.0238, "N": 0.0427, "P": 0.0469, "Q": 0.0393, "R": 0.0526,
                         "S": 0.0694, "T": 0.0550, "V": 0.0667, "W": 0.0118, "Y": 0.0311}
        self.aa_log_prior = {k: np.log(v) for k, v in self.aa_prior.items()}
        self.mem = {}
        
    def prob(self, seq):
        if seq in self.mem:
            return self.mem[seq]
        else:
            res = 1.
            for i in seq:
                res *= self.aa_prior[i]
            self.mem[seq] = res
            return res
        
    def log_prob(self, seq):
        if seq in self.mem:
            return self.mem[seq]
        else:
            res = 0.
            for i in seq:
                res += self.aa_log_prior[i]
            self.mem[seq] = res
            return res
        

position_mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': -4, '7': -3, '8': -2, '0': -1}
        
def extract_acg(seq, mask):
    return "".join(seq[position_mapping[j]] for j in mask)

def extract_acg2(seq, mask):
    return "".join(seq[j] for j in mask)

def nbscore(train_peptides, test_peptides, n_pan=10, n_spec=10, 
            pan_dac_name_candidates=['20', '12', '56', '23', '10', '80', '35', '70', '30', '24', '13'],
           ):

    pan_dac_names = pan_dac_name_candidates[:n_pan]
    
    train_data = pd.DataFrame({'Peptides': train_peptides})
    test_data = pd.DataFrame({'Peptides': test_peptides})
    
    train_data['length'] = train_data['Peptides'].apply(len)
    train_data = train_data[(train_data['length'] <= 11) & (train_data['length'] >= 8)]
    train_data = train_data.drop_duplicates()

    # Extract important locations
    # Select dacs
    dac_names = []
    for i, a in enumerate(position_mapping):
        for j, b in enumerate(position_mapping):
            if i < j:
                dac_names.append(a + b)

    for i in position_mapping:
        # print(train_data['Peptides'])
        train_data[i] = train_data['Peptides'].str[position_mapping[i]]

    for i in dac_names:
        mask = [position_mapping[j] for j in i]
        train_data[i] = train_data['Peptides'].apply(lambda x: extract_acg2(x, mask))


    entropy = pd.Series(index=list(position_mapping.keys()) + dac_names, dtype='float64')
    for i in position_mapping:
        entropy[i] = scipy.stats.entropy(train_data[i].value_counts())

    for i in dac_names:
        entropy[i] = scipy.stats.entropy(train_data[i].value_counts())

    mutual_info = pd.Series(index=dac_names, dtype='float64')
    for i in dac_names:
        mutual_info[i] = sum(entropy[j] for j in i) - entropy[i]

    acg_names = pan_dac_names.copy()
    cnt = n_spec
    for i in mutual_info.sort_values(ascending=False).index:
        if i not in acg_names:
            acg_names.append(i)
            cnt -= 1
        if cnt <= 0:
            break

    print(acg_names)
    
    log_total_count = np.log(train_data.shape[0])
    train_pos_log_probs = {i: np.log(train_data[i].value_counts()) - log_total_count for i in position_mapping}
    train_acg_log_probs = {i: np.log(train_data[i].value_counts()) - log_total_count for i in acg_names}


    for i in position_mapping:
        test_data[i] = test_data['Peptides'].str[position_mapping[i]]

    for i in acg_names:
        mask = [position_mapping[j] for j in i]
        test_data[i] = test_data['Peptides'].apply(lambda x: extract_acg2(x, mask))


    test_data['pos_prob'] = 0.
    for i in position_mapping:
        test_data['pos_prob'] += test_data[i].apply(lambda x: train_pos_log_probs[i].get(x, -2 - log_total_count))

    for i in acg_names:
        test_data['pos_prob'] += test_data[i].apply(lambda x: train_acg_log_probs[i].get(x, -2 - log_total_count))


    seq_prior = Seq_Prior()
    test_data['neg_prob'] = 0.
    for i in position_mapping:
        test_data['neg_prob'] += test_data[i].apply(seq_prior.log_prob)

    for i in acg_names:
        test_data['neg_prob'] += test_data[i].apply(seq_prior.log_prob)

    test_data['Score'] = test_data['pos_prob'] - np.logaddexp(test_data['pos_prob'], test_data['neg_prob'] + np.log(9999))
    return test_data['Score'].tolist()


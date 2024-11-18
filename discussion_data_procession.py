# Datasets preprocessing
import os

import numpy as np
import pandas as pd

from create_data import smile_to_graph, prot_to_graph, seq_cat
from discussion_utils import TestbedDataset

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
is_seq_in_graph = True
is_con_in_graph = True
is_profile_in_graph = True
is_emb_in_graph = True

compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []
opts = ['train', 'test']
dataset = ['discussion'][0]
# for opt in opts:
df = pd.read_csv('data/' + dataset + '/discussion.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
pdbs += list(df['target_name'])
pdbs_seqs += list(df['target_sequence'])
all_labels += list(df['affinity'])
# pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

# smile
saved_smile_graph = {}
for smile, pdb in set(zip(compound_iso_smiles, pdbs)):
    print('*'*10, pdb, '*'*10)
    g = smile_to_graph(smile)
    saved_smile_graph[smile] = g

# protein
saved_prot_graph = {}
# dataset = ['kiba', 'davis'][1]
for seq, t_name in set(zip(pdbs_seqs, pdbs)):
    if os.path.isfile('data/' + dataset + '/map/' + t_name + '.npy'):
        contactmap = np.load('data/' + dataset + '/map/' + t_name + '.npy')
    else:
        print(dataset, "-"*10, t_name, "-"*10)
        raise FileNotFoundError
    g2 = prot_to_graph(seq, contactmap, t_name, dataset)
    saved_prot_graph[t_name] = g2
# Datasets graph processing ends

# datasets = ['davis','kiba']
# convert to PyTorch data format
# for dataset in datasets:
processed_data_file_discussion = 'data/processed/' + dataset + '.pt'

if not os.path.isfile(processed_data_file_discussion):
    df = pd.read_csv('data/' + dataset + '/discussion.csv')
    discussion_drugs, discussion_prots, discussion_target, discussion_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['target_name']), list(df['affinity'])
    XT = [seq_cat(t) for t in discussion_prots]
    discussion_drugs, discussion_prots, discussion_target, discussion_Y = np.asarray(discussion_drugs), np.asarray(XT), np.asarray(discussion_target), np.asarray(discussion_Y)

    # make data PyTorch Geometric ready
    # print(smile_graph, "-" * 10, protein_graph, "-" * 10)
    print('preparing ', dataset + '.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset, xd=discussion_drugs, xt=discussion_prots, xtname=discussion_target, y=discussion_Y, smile_graph=saved_smile_graph, protein_graph=saved_prot_graph)
    print(processed_data_file_discussion, ' have been created')
else:
    print(processed_data_file_discussion, ' are already created')
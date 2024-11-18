import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def aa_ss_feature(target, dataset='davis'):
    feature = []
    file = 'data/'+dataset+'/profile/' + target + '_PROP/' + target + '.ss8'
    for line in open(file):
        cols = line.strip().split()
        if len(cols) == 11:
            res_sas = []
            res_sas.append(cols[-8])
            res_sas.append(cols[-7])
            res_sas.append(cols[-6])
            res_sas.append(cols[-5])
            res_sas.append(cols[-4])
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)

def aa_sas_feature(target, dataset='davis'):
    feature = []
    file = 'data/'+dataset+'/profile/' + target + '_PROP/' + target + '.acc'
    for line in open(file):
        if line[0] == '#':
            continue
        cols = line.strip().split()
        if len(cols) == 6:
            res_sas = []
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)

def aa_features(aa):
    results = one_of_k_encoding(aa,
                                ['A', 'N', 'C', 'Q', 'H', 'L', 'M', 'P', 'T', 'Y', 'R', 'D', 'E', 'G', 'I', 'K', 'F',
                                 'S', 'W', 'V', 'X'])
    return np.asarray(results, dtype=float)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def prot_to_graph(seq, prot_contactmap, prot_target, dataset='davis'):
    c_size = len(seq)
    # eds_seq = []
    # if is_seq_in_graph:
    #     for i in range(c_size - 1):
    #         eds_seq.append([i, i + 1])
    #     eds_seq = np.array(eds_seq)
    eds_contact = []
    if is_con_in_graph:
        eds_contact = np.array(np.argwhere(prot_contactmap >= 0.5))

    # add an reserved extra node for drug node
    # eds_d = []
    # for i in range(c_size):
    #     eds_d.append([i, c_size])

    # eds_d = np.array(eds_d)
    # if is_seq_in_graph and is_con_in_graph:
    #     eds = np.concatenate((eds_seq, eds_contact, eds_d))
    # elif is_con_in_graph:
    #     eds = np.concatenate((eds_contact, eds_d))
    # else:
    #     eds = np.concatenate((eds_seq, eds_d))

    # edges = [tuple(i) for i in eds]
    # g = nx.Graph(edges).to_directed()
    features = []
    ss_feat = []
    sas_feat = []
    if is_profile_in_graph:
        ss_feat = aa_ss_feature(prot_target, dataset)
        sas_feat = aa_sas_feature(prot_target, dataset)
    # sequence_output = np.load('data/'+dataset+'/emb/' + prot_target + '.npz', allow_pickle=True)
    # sequence_output = sequence_output[prot_target].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    # # sequence_output = pad_2d_unsqueeze(sequence_output, max_seq_len)
    # sequence_output = sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1])
    '''for i in range(c_size):
        if is_profile_in_graph:
            if is_emb_in_graph:
                aa_feat = np.concatenate((ss_feat[i], sas_feat[i]))
            else:
                aa_feat = np.concatenate((aa_features(seq[i]), ss_feat[i], sas_feat[i]))
        else:
            if is_emb_in_graph:
                aa_feat = np.asarray(eds_contact, dtype=float)
            else:
                aa_feat = aa_features(seq[i])
        features.append(aa_feat.tolist())
'''
    # place holder feature vector for drug
    # place_holder = np.zeros(features[0].shape, dtype=float)
    # features.append(place_holder)
    return c_size, ss_feat, sas_feat, eds_contact

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


# from DeepDTA data
all_prots = []
datasets = ['kiba', 'davis']
for dataset in datasets:
    print('\nconvert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    target = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
        target.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_name,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [target[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str,ls)) + '\n')       
    print('dataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))
# Datasets generation completed

# Datasets preprocessing
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
dataset = ['kiba', 'davis'][1]
for opt in opts:
    df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
    pdbs += list(df['target_name'])
    pdbs_seqs += list(df['target_sequence'])
    all_labels += list(df['affinity'])
# pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

# smile
saved_smile_graph = {}
for smile in set(compound_iso_smiles):
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
processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_target, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['target_name']), list(df['affinity'])
    XT = [seq_cat(t) for t in train_prots]
    train_drugs, train_prots, train_target, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_target), np.asarray(train_Y)
    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots, test_target, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['target_name']), list(df['affinity'])
    XT = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots, test_target, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_target), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    # print(smile_graph, "-" * 10, protein_graph, "-" * 10)
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, xtname=train_target, y=train_Y, smile_graph=saved_smile_graph, protein_graph=saved_prot_graph)
    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, xtname=test_target, y=test_Y, smile_graph=saved_smile_graph, protein_graph=saved_prot_graph)
    print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')

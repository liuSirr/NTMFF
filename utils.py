import os
import pickle

import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, xtname=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, protein_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xtname, y, smile_graph, protein_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, xtname, y, smile_graph, protein_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        # print(len(xd), '-'*10, len(xt), '-'*10, len(y), '-'*10, len(xtname))
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            target_name = xtname[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # print("-"*10, target_name, "-"*10)
            p_size, ss_feat, sas_feat, eds_contact = protein_graph[target_name]
            ss_feat = np.where(np.isclose(ss_feat, 0.), 0, 1)
            sas_feat = np.where(np.isclose(sas_feat, 0.), 0, 1)
            eds_contact = np.where(np.isclose(eds_contact, 0.), 0, 1)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                sas_feat=torch.FloatTensor([sas_feat]).squeeze(0),
                                y=torch.FloatTensor([labels]),
                                target=torch.LongTensor([target]),
                                c_size=torch.LongTensor([c_size]),
                                ss_feat=torch.FloatTensor([ss_feat]).squeeze(0),
                                eds_contact=torch.FloatTensor([eds_contact]).squeeze(0)
                                )
            # GCNData.target = torch.LongTensor([target])
            # GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # GCNData.__setitem__('p_size', torch.LongTensor([p_size]))
            # GCNData.__setitem__('aa_feat', torch.FloatTensor([ss_feat]).squeeze(0))
            # GCNData.__setitem__('sas_feat', torch.FloatTensor([sas_feat]).squeeze(0))
            # GCNData.__setitem__('eds_contact', torch.FloatTensor([eds_contact]).squeeze(0))

            # GCNData.target_edge_weight = torch.LongTensor([p_edge_weight])
            # GCNData.target_edge_index = torch.LongTensor([p_edge_index]).squeeze(0)
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        # print(data_list)
        #
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    if len(y) != len(f):
        f = f[0:len(y)]
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    if len(y) != len(f):
        f = f[0:len(y)]
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    if len(y) != len(f):
        f = f[0:len(y)]
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    if len(y) != len(f):
        f = f[0:len(y)]
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    if len(y) != len(f):
        f = f[0:len(y)]
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x

def pad_1d_unsqueeze(x, padlen):
    x = x  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x
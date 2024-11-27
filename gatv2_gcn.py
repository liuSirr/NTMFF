import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from msa import *
from torch_geometric.nn import GCNConv, global_add_pool, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model
from utils import pad_2d_unsqueeze


class GATv2_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2,
                 ss_feature=8, sas_feature=3, map_feature=2):

        super(GATv2_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATv2Conv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.fc_g3 = torch.nn.Linear(ss_feature*2, output_dim)
        self.fc_g4 = torch.nn.Linear(sas_feature * 2, output_dim)
        self.fc_g5 = torch.nn.Linear(map_feature * 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, 64, 1, dropout=0.5, bidirectional=True)
        self.msa = MheadsAtt(128, hid_dim=128, n_heads=2, pf_dim=128)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128 * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target, ss_feat, sas_feat, eds_contact = data.target, data.ss_feat, data.sas_feat, data.eds_contact
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # ss_feat
        temp_batch = torch.LongTensor(np.ones(ss_feat.size(0)) * (512-1)).to("cuda:0")
        ss_feat = torch.cat([gmp(ss_feat, temp_batch), gap(ss_feat, temp_batch)], dim=1)
        ss_feat = self.fc_g3(ss_feat)

        # sas_feat
        temp_batch = torch.LongTensor(np.ones(sas_feat.size(0)) * (512 - 1)).to("cuda:0")
        sas_feat = torch.cat([gmp(sas_feat, temp_batch), gap(sas_feat, temp_batch)], dim=1)
        sas_feat = self.fc_g4(sas_feat)

        # eds_contact
        temp_batch = torch.LongTensor(np.ones(eds_contact.size(0)) * (512 - 1)).to("cuda:0")
        eds_contact = torch.cat([gmp(eds_contact, temp_batch), gap(eds_contact, temp_batch)], dim=1)
        eds_contact = self.fc_g5(eds_contact)

        # target
        embedded_xt = self.embedding_xt(target)

        embedded_xt, _ = self.msa(embedded_xt, x)

        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        if len(x) < len(ss_feat) or len(xt) < len(ss_feat):
            x = pad_2d_unsqueeze(x, len(ss_feat))
            xt = pad_2d_unsqueeze(xt, len(ss_feat))

        xc = torch.cat((x, xt, ss_feat, sas_feat, eds_contact), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class MheadsAtt(nn.Module):
    def __init__(self, in_feats, hid_dim=128, n_heads=2, pf_dim=128):
        super(MheadsAtt, self).__init__()
        self.init_transform = nn.Linear(in_feats, hid_dim, bias=False)
        self.fc = nn.Linear(32, 128, bias=False)
        self.bn = nn.BatchNorm1d(hid_dim)
        self.attHeads = Decoder(128, hid_dim, 3, n_heads, pf_dim, DecoderLayer, SelfAttention,
                                PositionwiseFeedforward, 0.0)

    def forward(self, v_p, v_d):
        fatt = self.attHeads(v_p, v_d)
        return fatt
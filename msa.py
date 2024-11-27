## Inspired by: https://github.com/lifanchen-simm/transformerCPI
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 自注意模块输入 q k v
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)


        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        if len(query.shape)>len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        Q = query.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = key.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = value.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy =torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # Q,K = Q.cpu(),K.cpu()
        del Q, K
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        selfresult = self.fc(torch.matmul(self.do(F.softmax(energy, dim=-1)), V).permute(0, 2, 1, 3).contiguous().view(bsz, self.n_heads * (self.hid_dim // self.n_heads)))
        return selfresult

# 3 一维卷积输入一个x
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.do(x)
        x = self.fc_2(x)
        x = x
        return x

#  2 将自注意类和卷积当做参数传入
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        # q k v
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.ea = self_attention(hid_dim, n_heads, dropout)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src):
        trg1 = self.ln(trg + self.do(self.sa(trg, src, trg+src)))
        src1 = self.ln(src + self.do(self.sa(src, trg, src+trg)))
        return trg1, src1
# 1
class Decoder(nn.Module):
    """ 复合特征提取     decoder_layer：第二层；self_attention：第一层；positionwise_feedforward第三层  """
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                  positionwise_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout)
             for _ in range(n_layers)])
        self.init_line = nn.Linear(128, 256, bias=False)

    def forward(self, trg, src):
        for layer in self.layers:
            #  靶标、药物的嵌入输入，一层代表一次interformer
            trg, src = layer(trg, src)
        result = torch.cat((trg, src), dim=1)
        return result



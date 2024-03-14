import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

import torch.nn.utils.rnn as rnn

import numpy as np
import torch.nn.init as init
from gcnconv import GCNConvKeyNode
from embedding import WordEmbedding, Vocab


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

"""
EPS = 1e-8
NUM_METRIC = 4
class ImportanceEmbeddings(nn.Module):
    def __init__(self, config, device):
        super(ImportanceEmbeddings, self).__init__()
        self.hidden_size = config.ext_hidden_size
        self.importance_hidden_size = config.ext_hidden_size//NUM_METRIC
        self.importance_embedding_list = [nn.Embedding(config.importance_size, self.importance_hidden_size) for i in range(NUM_METRIC)]
        self.device = device

        self.LayerNorm = nn.LayerNorm(config.ext_hidden_size)
        self.dropout = nn.Dropout(config.ext_dropout)
        for i in range(NUM_METRIC):
            self.importance_embedding_list[i].to(self.device)

    def forward(self, importance):
        importance_embeddings = torch.zeros(importance.size(0), self.hidden_size)
        importance_embeddings = importance_embeddings.to(self.device)

        for i in range(NUM_METRIC):
            importance_embeddings[:, i*self.importance_hidden_size:(i+1)*self.importance_hidden_size] \
                = self.importance_embedding_list[i](importance[:, i])
        importance_embeddings = self.LayerNorm(importance_embeddings)
        importance_embeddings = self.dropout(importance_embeddings)
        return importance_embeddings
"""


class GCN(nn.Module):
    def __init__(self, args, device, embed=None, load_embedding=True, conv_class=pyg.nn.conv.gcn_conv.GCNConv):
        super(GCN, self).__init__()
        if args.encoder == 'gcn':
            self.gc0 = nn.Linear(args.embed_size, args.hidden_size)
            self.gc1 = conv_class(args.hidden_size, args.hidden_size, False)
            if args.auto_encoder == 'gae':
                self.gc2 = conv_class(args.hidden_size, args.hidden_size, False)
                self.ret_type = 'gae'
            else: #if args.auto_encoder == 'vgae':
                self.gc_mu = conv_class(args.hidden_size, args.hidden_size, False)
                self.gc_var = conv_class(args.hidden_size, args.hidden_size, False)
                self.ret_type = 'vgae'
            self.hidden_size = args.hidden_size 
            self.dropout = args.dropout

        self.load_vocab(args, device, load_embedding, embed)
    
    def load_vocab(self, args, device, load_embedding, embed=None):
        vocab = Vocab(args.vocab_path, args.vocab_size)
        self.device = device
        self.embed = embed if embed else torch.nn.Embedding(vocab.size(), args.embed_size, padding_idx=0)
        self.sent_embed = SentEncoder(args, self.embed, device)
        if load_embedding: 
            embed_loader = WordEmbedding(args.embedding_path, vocab)
            vectors = embed_loader.load_my_vecs(args.embed_size)
            pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.embed_size)
            self.embed.weight.data.copy_(torch.Tensor(pretrained_weight))
            # self.embed.weight.requires_grad = args.embed_train

    def concat_sent_rep(self, x, sent_x, embed=None):
        if embed is None:
            embed = self.embed(x)
        x = embed
        sent_x = self.sent_embed(x, sent_x)
        #print("compare embedding of word and sentence:\n", x, sent_x)
        x = torch.cat([x, sent_x], dim=1) # dim for sequence length
        x = self.gc0(x).view(-1, self.hidden_size)
        return x, embed

    def forward(self, x, sent_x, edge_index, embed=None):#, edge_weight):
        x, embed = self.concat_sent_rep(x, sent_x, embed)

        x = F.elu(self.gc1(x, edge_index)) #, edge_weight))
        x = F.dropout(x, self.dropout, training = self.training)
        if self.ret_type == 'gae':
            return self.gc2(x, edge_index), embed#, edge_weight) # [n_nodes, hidden_size]
        else: # if self.ret_type == 'vgae':
            return self.gc_mu(x, edge_index), self.gc_var(x, edge_index), embed


class GCNII(GCN):
    """GCN2, 加入残差机制"""
    def __init__(self, args, device, nlayers, dropout, theta, alpha, embed=None, load_embedding=True, variant=False,
                conv_class=pyg.nn.conv.gcn2_conv.GCN2Conv):
        super(GCNII, self).__init__(args, device, embed, load_embedding, conv_class)
        self.gc0 = nn.Linear(args.embed_size, args.hidden_size)
        if args.auto_encoder == 'gae':
            self.gc2 = conv_class(args.hidden_size, alpha, theta, nlayers)
            self.ret_type = 'gae'
        else: #if args.auto_encoder == 'vgae':
            self.gc_mu = conv_class(args.hidden_size, alpha, theta, nlayers)
            self.gc_var = conv_class(args.hidden_size, alpha, theta, nlayers)
            self.ret_type = 'vgae'
        self.hidden_size = args.hidden_size 
        self.dropout = args.dropout

        
        self.convs = nn.ModuleList()
        for ilayer in range(nlayers):
            self.convs.append(conv_class(args.hidden_size, alpha, theta, ilayer+1))
        self.fc = nn.Linear(args.hidden_size, args.hidden_size)
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, sent_x, edge_index):
        x = self.concat_sent_rep(x, sent_x)

        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fc(x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, _layers[0], edge_index))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        # return layer_inner
        if self.ret_type == 'gae':
            return self.gc2(layer_inner, _layers[0], edge_index) # [n_nodes, hidden_size]
        else: # if self.ret_type == 'vgae':
            return self.gc_mu(layer_inner, _layers[0], edge_index), \
                self.gc_var(layer_inner, _layers[0], edge_index)


class JKNet(nn.Module):
    """JKNet, 加入残差机制"，所有层和最终建立连接"""
    def __init__(self, nfeat, nlayers, nhidden, dropout, mode):
        super(JKNet, self).__init__()

        self.mode = mode.lower()
        assert self.mode in ['cat', 'max']
        if self.mode == 'cat':
            assert nfeat % nlayers == 0
            nhidden = nhidden // nlayers

        conv_class = pyg.nn.conv.gcn_conv.GCNConv
        self.convs = nn.ModuleList()
        for ilayer in range(nlayers-1):
            self.convs.append(conv_class(nhidden, nhidden, False))
        self.fc = nn.Linear(nfeat, nhidden)
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fc(x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = self.act_fn(con(layer_inner, edge_index))
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            _layers.append(layer_inner)

        if self.mode == 'cat':
            return torch.cat(_layers, dim=-1)
        elif self.mode == 'max':
            return torch.stack(_layers, dim=-1).max(dim=-1)[0]


class BiEncoder(nn.Module):
    """引入两种传播机制的Encoder"""
    def __init__(self, args, encoder1, encoder2):
        super(BiEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.ret_type = args.auto_encoder
    
    def forward(self, x, sent_x, edge_index, embed):
        if embed is None: 
            embed1, embed2 = None, None
        else:
            embed1, embed2 = embed[0], embed[1]
        if self.ret_type == 'gae':
            x1, embed1 = self.encoder1(x, sent_x, edge_index, embed1)
            x2, embed2 = self.encoder2(x, sent_x, edge_index, embed2)
            return torch.cat([x1, x2], dim=-1), (embed1, embed2)
        else: # if self.ret_type == 'vgae':
            mu1, var1, embed1 = self.encoder1(x, sent_x, edge_index, embed1)
            mu2, var2, embed2 = self.encoder2(x, sent_x, edge_index, embed2)
            return torch.cat([mu1, mu2], dim=-1), torch.cat([var1, var2], dim=-1), (embed1, embed2)


class GraphAttention(nn.Module):
    """计算注意力权重，用于合并不同部分的embeding"""
    def __init__(self, hidden_size):
        super(GraphAttention, self).__init__()
        in_size = hidden_size
        #hidden_size = 16
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.output_file = open("attentions2.txt", "w")

    def forward(self, z_list, clss=None):
        #z_list is a list of tensor with same shape
        z = torch.stack(z_list, dim=1)
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        if clss is not None:
            batch_size = z_list[0].size(0)
            output_beta = beta[clss]
            output_beta = beta.mean(0).squeeze(0).squeeze(-1)
            self.output_file.write("{}\n".format(output_beta.cpu().detach().numpy()))
            self.output_file.flush()
        return (beta * z).sum(1)

WORD_PAD = "[PAD]"


class SentEncoder(nn.Module):
    """LSTM+CNN生成句子embedding"""
    def __init__(self, hps, embed, device):
        """
        :param hps: 
                word_emb_dim: word embedding dimension
                sent_max_len: max token number in the sentence
                word_embedding: bool, use word embedding or not
                embed_train: bool, whether to train word embedding
                cuda: bool, use cuda or not
        """
        super(SentEncoder, self).__init__()

        self._hps = hps
        self.device = device
        embed_size = hps.embed_size

        input_channels = 1
        out_channels = 50
        min_kernel_size = 2
        max_kernel_size = 7
        width = embed_size

        # word embedding
        self.embed = embed

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(hps.max_pos + 1, embed_size, padding_idx=0), freeze=True)
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(hps.max_pos + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size=(height, width)) for height in
                                    range(min_kernel_size, max_kernel_size + 1)])
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))
        self.cnn_proj = nn.Linear(embed_size, self._hps.hidden_size)
        
        self.lstm = nn.LSTM(self._hps.hidden_size, self._hps.hidden_size, num_layers=2, dropout=0.1,
                            batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(self._hps.hidden_size * 2, self._hps.hidden_size)

    def cnn_forward(self, input, enc_embed_input, input_sent_len):
        # input: a batch of Example object [s_nodes, seq_len]
        sent_pos_list = []
        sent_max_len = input.size(-1)
        
        for sentlen in input_sent_len:
            sent_pos = list(range(1, min(sent_max_len, sentlen) + 1))
            sent_pos.extend([0] * int(sent_max_len - sentlen))
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long().to(self.device)#.cuda()

        enc_pos_embed_input = self.position_embedding(input_pos.long())  # [s_nodes, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1)  # [s_nodes, 1, L, D]
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs]  # kernel_sizes * [s_nodes, Co=50, W]
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output]  # kernel_sizes * [s_nodes, Co=50]
        sent_embedding = torch.cat(enc_maxpool_output, 1)  # [s_nodes, 50 * 6]
        return sent_embedding   # [s_nodes, 300]
    
    def _sent_cnn_feature(self, input, input_emb, snode_len, snode_pos):
        ngram_feature = self.cnn_forward(input, input_emb, snode_len)  # [snode, embed_size]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)

        lstm_embedding = torch.zeros(features.size(0), features.size(1), unpacked[0].size(-1)).to(self.device)
        for i in range(len(unpacked)):
            lstm_embedding[i][:unpacked_len[i]] = unpacked[i][:unpacked_len[i]]
        lstm_feature = self.lstm_proj(lstm_embedding)  # [n_nodes, n_feature_size]
        return lstm_feature

    def forward(self, x, sent_x):
        # node feature
        batch_size, sent_num, sent_length = sent_x.size(0), sent_x.size(1), sent_x.size(2)

        snode_pos = torch.LongTensor(list(range(sent_num))*batch_size).to(self.device)
        #.cuda() #CUDA device
        input_sent_len = (sent_x != 0).sum(dim=-1).int()  # [s_nodes, 1]
        sent_emb_x = self.embed(sent_x) # [s_nodes, L, D]
        cnn_feature = self._sent_cnn_feature(sent_x.view(batch_size*sent_num, -1), \
                sent_emb_x.view(batch_size*sent_num, sent_length, -1), \
                input_sent_len.view(batch_size*sent_num, -1), snode_pos)
        
        input_sent_len = (sent_x[:, :, 0] != 0).sum(dim=-1).int().cpu()  # [n_batch]
        cnn_feature = cnn_feature.view(batch_size, sent_num, -1)
        lstm_feature = self._sent_lstm_feature(cnn_feature , input_sent_len)
        
        sent_x = torch.cat([cnn_feature, lstm_feature], dim=-1)  # [n_nodes, n_feature_size * 2]
        return sent_x


def testSentEncoder(args, device):
    x = torch.rand(8, 512)*10000
    x = x.long().to(device)
    sent_x = torch.rand(8, 40, 50)*10000
    sent_x = sent_x.long().to(device)
    embed = torch.nn.Embedding(50000, args.embed_size, padding_idx=0).to(device)
    model = SentEncoder(args, embed, device)
    y = model(x, sent_x)
    
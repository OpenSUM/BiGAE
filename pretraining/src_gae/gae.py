from tkinter import N
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
)
from embedding import WordEmbedding, Vocab
from encoder import SentEncoder

import collections

EPS = 1e-15
MAX_LOGSTD = 10


def reset(nn):
    """更新参数"""
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class InnerProductDecoder(torch.nn.Module):
    """向量内积，作为GAE的Decoder"""
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    """图自编码器"""
    def __init__(self, args, encoder, device, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.loss_lambda = args.loss_lambda
        if args.pre_loss in ['2ce', 'summ_ce']:
            self.trans = torch.nn.Linear(args.hidden_size, args.hidden_size)
        if args.pre_loss == 'mse':
            self.mse =  torch.nn.MSELoss(reduction='mean')
        self.device = device
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), 
                                            num_neg_samples=pos_edge_index.size(1)*2)
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return self.loss_lambda * pos_loss + (1 - self.loss_lambda) * neg_loss
    
    def summ_recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Predict whether the xxx is xxx
        """
        z = self.trans(z)
        return self.recon_loss(z, pos_edge_index, neg_edge_index)
    
    def mse_loss(self, z, pos_edge_index, pos_edge_attr, neg_edge_index=None):
        """MSE Loss作为目标函数

        Args:
            z : 节点embedding
            pos_edge_index : 摘要节点index
            pos_edge_attr : 预测值
            neg_edge_index : 非摘要节点index

        Returns:
            Loss函数
        """
        p_edge = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = self.mse(p_edge.float(), pos_edge_attr.float())  

        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), p_edge.size(0))
        
        p_neg_edge = self.decoder(z, neg_edge_index, sigmoid=True)
        neg_loss = self.mse(p_neg_edge.float(), torch.zeros_like(p_neg_edge).to(self.device).float())
        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    "Variational Graph Auto Encoder"
    def __init__(self, args, encoder, device, decoder=None):
        super().__init__(args, encoder, device, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__, embed = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, embed

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class AutoEncoder(torch.nn.Module):
    """集成了GAE,VGAE以及不同loss函数的AutoEncoder"""
    def __init__(self, args, encoder, device):
        super().__init__()
        self.ret_type = args.auto_encoder
        self.add_klloss = args.klloss
        self.pre_loss = args.pre_loss
        self.device = device
        if self.ret_type == "vgae":
            self.auto_encoder = VGAE(args, encoder, device)
        else:
            self.auto_encoder = GAE(args, encoder, device)
        
    def forward(self, src, graph_data, src_for_sent, embed=None, *args, **kwargs):
        """模型前馈

        Args:
            src : 原文
            graph_data : 构造的图数据
            src_for_sent : 用于LSTM的句子张量
            embed : 只有词节点的embedding

        Returns:
           loss 或者 包含loss 句子分数 embedding的dict
        """
        y, embed = self.auto_encoder.encode(src, src_for_sent, graph_data.edge_index, embed=embed) #获取词和句子的向量

        # 预测任务，mse:预测tf-idf值
        if self.pre_loss == 'ce':
            loss = self.auto_encoder.recon_loss(y, graph_data.edge_index) #self.loss(y, labels)
        elif self.pre_loss == '2ce':
            loss = self.auto_encoder.recon_loss(y, graph_data.edge_index) #self.loss(y, labels)
            summ_loss = self.auto_encoder.summ_recon_loss(y, graph_data.summ_edge_index)
            loss = loss + summ_loss
        elif self.pre_loss == 'summ_ce':
            summ_loss = self.auto_encoder.summ_recon_loss(y, graph_data.summ_edge_index)
        elif self.pre_loss == 'mse':
            loss = self.auto_encoder.mse_loss(y, graph_data.edge_index, graph_data.centrality) 
        
        if self.add_klloss and self.ret_type == "vgae":
            kl_loss = self.auto_encoder.kl_loss()
            loss = loss + kl_loss
        
        #TODO: 获取句子的向量，用以后续处理
        return {"loss": loss, "sent_scores":y, "embed": embed} 

    def forward_test(self, src, graph_data, src_for_sent, embed=None, *args, **kwargs):
        """测试时的前馈，返回正例负例的分类准确率"""
        z, embed = self.auto_encoder.encode(src, src_for_sent, graph_data.edge_index, embed)
        
        pos_edge_index = graph_data.edge_index
        p_pos = self.auto_encoder.decoder(z, pos_edge_index, sigmoid=True) + EPS #通过向量还原图
        
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        p_neg = 1 - self.auto_encoder.decoder(z, neg_edge_index, sigmoid=True) + EPS

        pos_diff = ((p_pos - graph_data.centrality).abs()).cpu()
        neg_diff = (p_neg).cpu()
        # print(p_pos.shape,graph_data.centrality.shape)
        # acc_pos = (p_pos > 0.5).float().mean().cpu()
        # acc_neg = (p_neg > 0.5).float().mean().cpu()
        # print('centrality')
        # print(p_pos.tolist())
        # print(p_neg.tolist())
        return {"pos_diff": pos_diff, "neg_diff": neg_diff}
        

class Summarizer(torch.nn.Module):
    """摘要生成的图神经网络模型"""
    def __init__(self, args, encoder, device, checkpoint=None):
        super().__init__()
        self.hidden_size = args.hidden_size*2 if args.conv == 'both' else args.hidden_size
        self.encoder = encoder
        self.encoder_freeze = args.encoder_freeze
        self.device = device

        # 选择层数不同的MLP
        if args.nlayer_cls == 1:
            self.cls = torch.nn.Linear(self.hidden_size, 1)
        elif args.nlayer_cls == 2:
            self.cls = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.hidden_size, 1))
        else: #if args.nlayer_cls > 2:
            cls_list = [('lin0', torch.nn.Linear(self.hidden_size, 1))]
            for i in range(1, args.nlayer_cls):
                cls_list = [('lin%d'%i, torch.nn.Linear(self.hidden_size, self.hidden_size)), \
                            ('relu%d'%i, torch.nn.ReLU())] + cls_list
            self.cls = torch.nn.Sequential(collections.OrderedDict(cls_list))

        self.loss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        if checkpoint is not None:
            self.load_state_dict(checkpoint)
        
        self.ret_type = args.auto_encoder
        
    def forward(self, src, graph_data, mask_cls, src_for_sent, embed=None, labels=None, **kwargs):
        """抽取摘要

        Args:
            src : 原文
            graph_data : 图数据
            mask_cls : 分类概率的mask
            src_for_sent : 句子id张量
            embed : 图节点embedding
            labels : 分类标签

        Returns:
            loss或包含loss、句子概率、节点embedding、句子embedding的dict
        """
        batch_size = src.size(0)
        sequence_length = src.size(1)
        if self.encoder_freeze: # 不更新encoder的参数
            with torch.no_grad():
                if self.ret_type == 'vgae':
                    y, _, embed = self.encoder(src, src_for_sent, graph_data.edge_index, embed=embed)
                else:
                    y, embed = self.encoder(src, src_for_sent, graph_data.edge_index, embed=embed)
        elif self.ret_type == 'vgae': # 更新encoder的参数
            mu, logstd, embed = self.encoder(src, src_for_sent, graph_data.edge_index, embed=embed)
            logstd = logstd.clamp(max=MAX_LOGSTD)
            if self.training:
                y = mu + torch.randn_like(logstd) * torch.exp(logstd)
            else: 
                y = mu
        else:
            y, embed = self.encoder(src, src_for_sent, graph_data.edge_index, embed=embed)
        y = y.view(batch_size, -1, self.hidden_size) #
        y = y[:, sequence_length:]
        sent_vec = y
        
        y = self.cls(y)

        logits = y.squeeze(-1)
        #logits = logits[torch.arange(batch_size).unsqueeze(1), clss]
        sent_scores = self.sigmoid(logits) * mask_cls.float()

        if labels is not None: # 若正在训练，返回Loss
            loss = self.loss(sent_scores, labels.float())
        return {"loss": loss if labels is not None else None, "sent_scores":sent_scores, "embed": embed, "sent_vec":sent_vec}


class MultiTaskSummarizer(torch.nn.Module):
    """自编码和摘要任务同时训练的模型"""
    def __init__(self, args, auto_encoder, device, checkpoint=None):
        super().__init__()
        self.hidden_size = args.hidden_size*2 if args.conv == 'both' else args.hidden_size
        self.auto_encoder = auto_encoder
        self.device = device
        if args.nlayer_cls == 1:
            self.cls = torch.nn.Linear(self.hidden_size, 1)
        elif args.nlayer_cls == 2:
            self.cls = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.hidden_size, 1))
        else: #if args.nlayer_cls > 2:
            cls_list = [('lin0', torch.nn.Linear(self.hidden_size, 1))]
            for i in range(1, args.nlayer_cls):
                cls_list = [('lin%d'%i, torch.nn.Linear(self.hidden_size, self.hidden_size)), \
                            ('relu%d'%i, torch.nn.ReLU())] + cls_list
            self.cls = torch.nn.Sequential(collections.OrderedDict(cls_list))

        self.loss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        if checkpoint is not None:
            self.load_state_dict(checkpoint)
        
        self.ret_type = args.auto_encoder
        
    def forward(self, src, graph_data, mask_cls, src_for_sent, labels=None, **kwargs):
        batch_size = src.size(0)
        sequence_length = src.size(1)
        ret = self.auto_encoder(src, graph_data, src_for_sent)
        y = ret["sent_scores"]
        y = y.view(batch_size, -1, self.hidden_size)
        y = y[:, sequence_length:]
        
        y = self.cls(y)

        logits = y.squeeze(-1)
        #logits = logits[torch.arange(batch_size).unsqueeze(1), clss]
        sent_scores = self.sigmoid(logits) * mask_cls.float()

        loss = ret["loss"] * 0.5
        if labels is not None:
            loss = loss + self.loss(sent_scores, labels.float())
        return {"loss": loss, "sent_scores":sent_scores}


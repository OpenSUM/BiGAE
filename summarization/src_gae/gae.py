from tkinter import N
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
)

import collections

EPS = 1e-15
MAX_LOGSTD = 10


def reset(nn):
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
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
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
    def __init__(self, args, encoder, device):
        super().__init__()
        self.hidden_size = args.hidden_size*2 if args.conv == 'both' else args.hidden_size
        self.ret_type = args.auto_encoder
        self.add_klloss = args.klloss
        self.pre_loss = args.pre_loss
        self.device = device
        if self.ret_type == "vgae":
            self.auto_encoder = VGAE(args, encoder, device)
        else:
            self.auto_encoder = GAE(args, encoder, device)
        
    def forward(self, src, graph_data, src_for_sent, embed=None, *args, **kwargs):
        y, embed = self.auto_encoder.encode(src, src_for_sent, graph_data.edge_index, embed=embed) #获取词和句子的向量
        sequence_length = src.size(1)

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
            loss = self.auto_encoder.mse_loss(y, graph_data.edge_index, graph_data.importance) 
        
        if self.add_klloss and self.ret_type == "vgae":
            kl_loss = self.auto_encoder.kl_loss()
            loss = loss + kl_loss
        
        #TODO: 获取句子的向量，用以后续处理
        batch_size = src.size(0)
        sent_vec = y.view(batch_size, -1, self.hidden_size) 
        sent_vec = sent_vec[:, sequence_length:]
        return {"loss": loss, "sent_scores":y, "embed": embed, "sent_vec": sent_vec} 

    def forward_test(self, src, graph_data, src_for_sent, embed=None, *args, **kwargs):
        z, embed = self.auto_encoder.encode(src, src_for_sent, graph_data.edge_index, embed)
        
        pos_edge_index = graph_data.edge_index
        p_pos = self.auto_encoder.decoder(z, pos_edge_index, sigmoid=True) + EPS
        
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        p_neg = 1 - self.auto_encoder.decoder(z, neg_edge_index, sigmoid=True) + EPS
        acc_pos = (p_pos > 0.5).float().mean().cpu()
        acc_neg = (p_neg > 0.5).float().mean().cpu()
        return {"acc_pos": acc_pos, "acc_neg": acc_neg, "acc": (acc_pos+acc_neg)/2}
        
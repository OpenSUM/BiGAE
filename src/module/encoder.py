import dgl
import dgl.udf
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np

from src.module.embedding import get_sinusoid_position_embedding
from src.utils import get_node_filter_fn, get_edge_filter_fn, get_word_node_filter_fn, get_sent_node_filter_fn


class GraphAttention(nn.Module):
    def __init__(
        self, 
        src_node_type_id: int, 
        dst_node_type_id: int, 
        edge_feat_name: str, 
        node_feat_size: int, 
        edge_feat_size: int, 
        out_size:int
    ):
        super(GraphAttention, self).__init__()

        self.src_node_type_id = src_node_type_id
        self.dst_node_type_id = dst_node_type_id
        self.edge_feat_name = edge_feat_name

        self.node_feat_proj = nn.Linear(node_feat_size, out_size, bias=False)
        self.edge_feat_proj = nn.Linear(edge_feat_size, out_size, bias=False)
        self.attn_fc = nn.Linear(3 * out_size, 1, bias=False)

        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def attn_func(self, edges: dgl.udf.EdgeBatch):
        edge_z = self.edge_feat_proj(edges.data[self.edge_feat_name])
        z = torch.cat((edges.src["z"], edges.dst["z"], edge_z), 1)

        attn_logits = self.act(self.attn_fc(z))

        return {"attn_logits": attn_logits}

    def message_func(self, edges: dgl.udf.EdgeBatch):
        return {"v": edges.src["z"], "attn_logits": edges.data["attn_logits"]}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        attn_scores = self.softmax(nodes.mailbox["attn_logits"])
        u = torch.sum(attn_scores * nodes.mailbox["v"], 1)
        return {"u": u}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        # Select the source and destination nodes, and corresponding edges
        src_nodes = g.filter_nodes(get_node_filter_fn(self.src_node_type_id))
        dst_nodes = g.filter_nodes(get_node_filter_fn(self.dst_node_type_id))
        edges = g.filter_edges(get_edge_filter_fn(self.src_node_type_id, self.dst_node_type_id))

        node_z = self.node_feat_proj(h)
        g.nodes[src_nodes].data["z"] = node_z
        # Get attention logits
        g.apply_edges(self.attn_func, edges=edges)
        g.pull(v=dst_nodes, message_func=self.message_func, reduce_func=self.reduce_func)

        g.ndata.pop("z")
        u = g.ndata.pop("u")

        return u[dst_nodes]


class MultiHeadGraphAttention(nn.Module):
    def __init__(
        self,
        src_node_type_id: int,
        dst_node_type_id: int,
        edge_feat_name: int,
        input_size: int,
        edge_feat_size: int,
        out_size: int,
        n_heads: int,
        dropout: float,
    ):
        super(MultiHeadGraphAttention, self).__init__()

        attn_params = {
            "src_node_type_id": src_node_type_id,
            "dst_node_type_id": dst_node_type_id,
            "edge_feat_name": edge_feat_name,
            "node_feat_size": input_size,
            "edge_feat_size": edge_feat_size,
            "out_size": out_size
        }

        self.heads = nn.ModuleList([GraphAttention(**attn_params) for _ in range(n_heads)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        out = [attn_head(g, self.dropout(h)) for attn_head in self.heads]
        # shape: [n_nodes, n_heads * out_size]
        out = torch.cat(out, 1)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w1 = nn.Conv1d(d_in, d_hid, 1)
        self.w2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class GATModel(nn.Module):
    def __init__(
        self,
        src_node_type_id: int,
        dst_node_type_id: int,
        edge_feat_name: str,
        input_size: int,
        edge_feat_size: int,
        ffn_hidden_size: int,
        out_size: int,
        n_heads: int,
        dropout: float
    ):
        super(GATModel, self).__init__()

        self.attn_layer = MultiHeadGraphAttention(
            src_node_type_id=src_node_type_id,
            dst_node_type_id=dst_node_type_id,
            edge_feat_name=edge_feat_name,
            input_size=input_size,
            edge_feat_size=edge_feat_size,
            out_size=out_size // n_heads,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.act = nn.ELU()
        self.ffn = PositionwiseFeedForward(out_size, ffn_hidden_size, dropout=dropout)

    def forward(self, g: dgl.DGLGraph, src_h: torch.Tensor, dst_h: torch.Tensor) -> torch.Tensor:
        residual = dst_h

        out = self.act(self.attn_layer(g, src_h))
        out = out + residual
        out = self.ffn(out.unsqueeze(0)).squeeze(0)

        return out


class TextCnn(nn.Module):
    def __init__(
        self,
        input_size: int,
        out_size: int,
        in_channels: int = 1,
        out_channels: int = 50,
        min_kernel_size: int = 2,
        max_kernel_size: int = 7,
    ):
        super(TextCnn, self).__init__()
        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(get_sinusoid_position_embedding(512, input_size), freeze=True)

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=(ks, input_size)) for ks in range(min_kernel_size, max_kernel_size + 1)]
        )

        self.fc = nn.Linear(out_channels * (max_kernel_size - min_kernel_size + 1), out_size)

        for conv in self.convs:
            init_weight_value = 6.0
            nn.init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))

    def forward(self, x, mask):
        # [snodes, max_len, word_dim]
        pos_inds = torch.arange(x.size(1)).to(x.device)
        word_pos_emb = self.position_embedding(pos_inds).unsqueeze(0)

        x = x + word_pos_emb
        x = x.masked_fill_(~mask.unsqueeze(-1), 0.0)
        # shape: [snodes, 1, max_len, word_dim]
        x = x.unsqueeze(1)

        # shape: [snodes, 50, max_len - ks + 1]
        outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # shape: [snodes, 50]
        outs = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in outs]

        # shape: [snodes, 6 * 50]
        o = torch.cat(outs, 1)

        c = self.fc(o)

        return c, o


class LstmEncoder(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        out_size: int, 
        n_layers: int, 
        dropout: float = 0.1, 
        is_bidirectional: bool = True
    ):
        super(LstmEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=is_bidirectional,
        )
        if is_bidirectional:
            self.out_proj = nn.Linear(hidden_size * 2, out_size)
        else:
            self.out_proj = nn.Linear(hidden_size, out_size)

    def forward(self, x, lens):
        # x shape: [n_graph, max_n_snode, dim]
        pad_x = rnn.pad_sequence(x, batch_first=True)
        x = rnn.pack_padded_sequence(pad_x, lens, batch_first=True)

        o, _ = self.lstm(x)

        o, o_lens = rnn.pad_packed_sequence(o, batch_first=True)

        # shape: n_graph * [n_snode, dim]
        o = [o[i][: o_lens[i]] for i in range(len(o))]

        # shape: [sum_n_node, dim]
        o = self.out_proj(torch.cat(o, 0))

        return o


class SentenceEncoder(nn.Module):
    def __init__(
        self, 
        word_emb_size: int, 
        lstm_hidden_size: int, 
        hidden_size: int, 
        out_size: int, 
        n_lstm_layers: int
    ):
        super(SentenceEncoder, self).__init__()
        self.textcnn = TextCnn(word_emb_size, hidden_size)
        self.lstm = LstmEncoder(word_emb_size, lstm_hidden_size, hidden_size, n_lstm_layers)
        self.out_proj = nn.Linear(2 * hidden_size, out_size)

    def forward(self, batched_graph: dgl.DGLGraph, word_encoder: nn.Embedding):
        sent_nodes = batched_graph.filter_nodes(get_sent_node_filter_fn())
        sent_ids = batched_graph.nodes[sent_nodes].data["ids"]

        word_embs_of_sent = word_encoder(sent_ids)
        mask = ~(sent_ids == 0)

        cnn_outs, ngram_feat = self.textcnn(word_embs_of_sent, mask)
        batched_graph.nodes[sent_nodes].data["ngram_feat"] = ngram_feat

        batch_size = batched_graph.batch_size
        ngram_feat_per_graph = []
        n_sents_per_graph = []
        for idx in range(batch_size):
            g = dgl.slice_batch(batched_graph, idx)
            snodes = g.filter_nodes(get_sent_node_filter_fn())
            ngram_feat_per_graph.append(g.nodes[snodes].data["ngram_feat"])
            n_sents_per_graph.append(len(snodes))

        lstm_outs = self.lstm(ngram_feat_per_graph, n_sents_per_graph)

        sent_emb = torch.cat((cnn_outs, lstm_outs), 1)
        sent_emb = self.out_proj(sent_emb)

        batched_graph.ndata.pop("ngram_feat")

        return sent_emb


class TopologyEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size):
        super(TopologyEncoder, self).__init__()
        self.encoders = [nn.Embedding(num_embeddings, embedding_size) for _ in range(4)]
        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        o = [encoder(x[:, i]) for i, encoder in enumerate(self.encoders)]
        o = torch.cat(o, 1)
        
        return o


class FusionAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x):
        logits = self.project(x)
        alpha = torch.softmax(logits, dim=1)

        o = torch.sum(alpha * x, dim=1)

        return o

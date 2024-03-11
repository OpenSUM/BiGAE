import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.udf
import pytorch_lightning as pl
import numpy as np
from torchcontrib.optim import SWA

from src.module.encoder import GATModel, SentenceEncoder, TopologyEncoder, FusionAttention
from src.module.vocabulary import Vocab
from src.module.embedding import get_pretrained_word_embedding, get_sinusoid_position_embedding
from src.module.metric import RougeMetricFromRouge, RougeMetricFromPyRouge
from src.utils import get_word_node_filter_fn, get_sent_node_filter_fn


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.requires_grad and (name.startswith("semantic_word_encoder") or name.startswith("semantic_edge_encoder") or name.startswith("topology_edge_encoder")):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.requires_grad and (name.startswith("semantic_word_encoder") or name.startswith("semantic_edge_encoder") or name.startswith("topology_edge_encoder")): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GraphSum(pl.LightningModule):
    def __init__(
        self,
        vocab_fp: str,
        word_emb_size: int,
        sent_emb_size: int,
        edge_feat_size: int,
        sent_hidden_size: int,
        lstm_hidden_size: int,
        n_lstm_layers: int,
        n_heads: int,
        dropout: float,
        ffn_hidden_size: int,
        n_iters: int,
        learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        weight_decay: float,
        max_n_sents: int,
        extract_n_sents: int,
    ):
        super(GraphSum, self).__init__()
        self.save_hyperparameters()

        self.vocab = Vocab(vocab_fp, 50000)
        
        self.semantic_word_encoder = nn.Embedding.from_pretrained(get_pretrained_word_embedding(self.vocab, word_emb_size), freeze=True)
        self.semantic_sent_encoder = SentenceEncoder(word_emb_size, lstm_hidden_size, sent_hidden_size, sent_emb_size, n_lstm_layers)
        self.topology_encoder = TopologyEncoder(10, word_emb_size // 4)
        self.topology_sent_proj = nn.Linear(word_emb_size, sent_emb_size)
        # self.topology_sent_encoder = TopologyEncoder(10, sent_emb_size // 4)
        self.position_sent_encoder = nn.Embedding.from_pretrained(get_sinusoid_position_embedding(max_n_sents, sent_emb_size), freeze=True)
        self.semantic_edge_encoder = nn.Embedding(10, edge_feat_size)
        self.topology_edge_encoder = nn.Embedding(10, edge_feat_size)

        self.word_to_sent_of_semantic = GATModel(src_node_type_id=0, dst_node_type_id=1,
                                                 edge_feat_name="semantic", input_size=word_emb_size, 
                                                 edge_feat_size=edge_feat_size, ffn_hidden_size=ffn_hidden_size, 
                                                 out_size=sent_emb_size, n_heads=n_heads, dropout=dropout)
        self.sent_to_word_of_semantic = GATModel(src_node_type_id=1, dst_node_type_id=0,
                                                 edge_feat_name="semantic", input_size=sent_emb_size, 
                                                 edge_feat_size=edge_feat_size, ffn_hidden_size=ffn_hidden_size, 
                                                 out_size=word_emb_size, n_heads=6, dropout=dropout)

        self.word_to_sent_of_topology = GATModel(src_node_type_id=0, dst_node_type_id=1,
                                                 edge_feat_name="semantic", input_size=word_emb_size, 
                                                 edge_feat_size=edge_feat_size, ffn_hidden_size=ffn_hidden_size, 
                                                 out_size=sent_emb_size, n_heads=n_heads, dropout=dropout)
        self.sent_to_word_of_topology = GATModel(src_node_type_id=1, dst_node_type_id=0,
                                                 edge_feat_name="semantic", input_size=sent_emb_size, 
                                                 edge_feat_size=edge_feat_size, ffn_hidden_size=ffn_hidden_size, 
                                                 out_size=word_emb_size, n_heads=6, dropout=dropout)

        self.n_iters = n_iters

        self.fusion_attn = FusionAttention(sent_emb_size, 16)
        self.classifier = nn.Linear(sent_emb_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_metric = RougeMetricFromRouge()
        self.test_metric = RougeMetricFromPyRouge() 

        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self.extract_n_sents = extract_n_sents

        self.fgm = FGM(self)

    @classmethod
    def from_hydra_args(cls, cfg):
        return cls(
            vocab_fp=cfg.vocab_fp,
            word_emb_size=cfg.word_emb_size,
            sent_emb_size=cfg.sent_emb_size,
            edge_feat_size=cfg.edge_feat_size,
            sent_hidden_size=cfg.sent_hidden_size,
            lstm_hidden_size=cfg.lstm_hidden_size,
            n_lstm_layers=cfg.n_lstm_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            ffn_hidden_size=cfg.ffn_hidden_size,
            n_iters=cfg.n_iters,
            learning_rate=cfg.learning_rate,
            total_steps=cfg.total_steps,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
            max_n_sents=cfg.max_n_sents,
            extract_n_sents=cfg.extract_n_sents,
        )

    def forward(self, batched_graph: dgl.DGLGraph):
        word_nodes = batched_graph.filter_nodes(get_word_node_filter_fn())
        sent_nodes = batched_graph.filter_nodes(get_sent_node_filter_fn())

        semantic_word_node_emb = self.semantic_word_encoder(batched_graph.nodes[word_nodes].data["id"])  # [n_word_nodes, word_emb_size]
        topology_word_node_emb = self.topology_encoder(batched_graph.nodes[word_nodes].data["centrality"])

        semantic_sent_node_emb = self.semantic_sent_encoder(batched_graph, self.semantic_word_encoder)
        topology_sent_node_emb = self.topology_encoder(batched_graph.nodes[sent_nodes].data["centrality"])
        topology_sent_node_emb = self.topology_sent_proj(topology_sent_node_emb)
        position_sent_node_emb = self.position_sent_encoder(batched_graph.nodes[sent_nodes].data["position"])

        batched_graph.edata["semantic"] = self.semantic_edge_encoder(batched_graph.edata["semantic"])
        # batched_graph.edata["topology"] = self.topology_edge_encoder(batched_graph.edata["topology"])


        semantic_word_hidden_state = semantic_word_node_emb
        semantic_sent_hidden_state = self.word_to_sent_of_semantic(
            batched_graph, semantic_word_hidden_state, semantic_sent_node_emb + position_sent_node_emb)

        topology_word_hidden_state = topology_word_node_emb
        topology_sent_hidden_state = self.word_to_sent_of_topology(
            batched_graph, topology_word_hidden_state, topology_sent_node_emb + position_sent_node_emb)

        for _ in range(self.n_iters):
            semantic_word_hidden_state = self.sent_to_word_of_semantic(
                batched_graph, semantic_sent_hidden_state, semantic_word_hidden_state)
            semantic_sent_hidden_state = self.word_to_sent_of_semantic(
                batched_graph, semantic_word_hidden_state, semantic_sent_hidden_state)

            topology_word_hidden_state = self.sent_to_word_of_topology(
                batched_graph, topology_sent_hidden_state, topology_word_hidden_state)
            topology_sent_hidden_state = self.word_to_sent_of_topology(
                batched_graph, topology_word_hidden_state, topology_sent_hidden_state)

        fin_sent_emb = torch.stack((semantic_sent_hidden_state, topology_sent_hidden_state), dim=1)
        fin_sent_emb = self.fusion_attn(fin_sent_emb)

        logits = self.classifier(fin_sent_emb).view(-1)

        return logits

    def training_step(self, batched_data, batch_idx):
        from copy import deepcopy
        copy_batched_data = deepcopy(batched_data)
        batched_graph: dgl.DGLGraph = batched_data.graph

        logits1 = self(batched_graph)
        logits2 = self(copy_batched_data.graph)

        sent_nodes = batched_graph.filter_nodes(get_sent_node_filter_fn())
        labels = batched_graph.nodes[sent_nodes].data["labels"].float()
        ce_loss = 0.5 * (self.loss_fn(logits1, labels) + self.loss_fn(logits2, labels))
        kl_loss = self.compute_kl_loss(logits1, logits2)
        loss = kl_loss * 0.5 + ce_loss

        self.log("loss/train", loss.item(), sync_dist=True)

        return {"loss": loss}

    # def training_step_end(self, training_step_outputs):
    #    loss = training_step_outputs["loss"]
    #    loss.backward()
    #
    #    self.fgm.attack()
    #    batched_data = training_step_outputs["batched_data"]
    #    loss = self.training_step(batched_data, None)["loss"]
    #
    #    return loss

    # def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
    #    loss.backward()
    #    self.fgm.restore()

    ############# SWA Code
    # def training_epoch_end(self, outputs):
    #     self.optimizers().update_swa()
    #     return

    # def on_train_end(self):
    #     self.optimizers().swap_swa_sgd()
    #     return

    def compute_kl_loss(self, p, q, pad_mask=None):
    
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def validation_step(self, batched_data, batch_idx):
        batched_graph: dgl.DGLGraph = batched_data.graph
        src_txt = batched_data.src_txt
        tgt_txt = batched_data.tgt_txt

        logits = self(batched_graph)

        sent_nodes = batched_graph.filter_nodes(get_sent_node_filter_fn())
        labels = batched_graph.nodes[sent_nodes].data["labels"].float()

        loss = self.loss_fn(logits, labels)
        batched_graph.nodes[sent_nodes].data["logits"] = logits

        hyps = self.extract_summ(batched_graph, src_txt)
        refer = ["\n".join(ref) for ref in tgt_txt]
        self.val_metric.update(hyps, refer)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        tl_val_loss = [item["loss"].item() for item in outputs]

        val_loss = np.mean(tl_val_loss)

        metrics = self.val_metric.compute()

        self.log("loss/valid", val_loss, sync_dist=True)
        self.log("val_rouge1", metrics["rouge1"], sync_dist=True, prog_bar=True)
        self.log("val_rouge2", metrics["rouge2"], sync_dist=True, prog_bar=True)
        self.log("val_rougel", metrics["rougel"], sync_dist=True, prog_bar=True)

    def test_step(self, batched_data, batch_idx):
        batched_graph: dgl.DGLGraph = batched_data.graph
        src_txt = batched_data.src_txt
        tgt_txt = batched_data.tgt_txt

        logits = self(batched_graph)

        sent_nodes = batched_graph.filter_nodes(get_sent_node_filter_fn())
        labels = batched_graph.nodes[sent_nodes].data["labels"].float()

        loss = self.loss_fn(logits, labels)
        batched_graph.nodes[sent_nodes].data["logits"] = logits

        hyps = self.extract_summ(batched_graph, src_txt)
        refer = ["\n".join(ref) for ref in tgt_txt]
        self.test_metric.update(hyps, refer)

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        metrics = self.test_metric.compute()

        self.log("test_rouge1", metrics["rouge1"], sync_dist=True)
        self.log("test_rouge2", metrics["rouge2"], sync_dist=True)
        self.log("test_rougel", metrics["rougel"], sync_dist=True)

    def extract_summ(self, batched_graph, src_txt):
        def _get_ngrams(n, text):
            """get n-gram of texts, borrowed from https://github.com/nlpyang/BertSum"""
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            """do 3-gram block, borrowed from https://github.com/nlpyang/BertSum"""
            tri_c = _get_ngrams(3, c.split())
            for i, s in enumerate(p):
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        batch_size = batched_graph.batch_size

        hyps = []
        for idx in range(batch_size):
            g = dgl.slice_batch(batched_graph, idx)

            sent_nodes = g.filter_nodes(get_sent_node_filter_fn())
            logits = g.nodes[sent_nodes].data["logits"]
            scores = logits.sigmoid()
            scores = scores.detach().cpu().numpy()

            ranks = np.argsort(scores)

            tmp = []
            for rk in ranks[::-1]:
                if rk >= len(src_txt[idx]):
                    continue

                cur_sent = src_txt[idx][rk].strip()

                if not _block_tri(cur_sent, tmp):
                    tmp.append(cur_sent)

                if len(tmp) == self.extract_n_sents:
                    break
            hyps.append("\n".join(tmp))

        return hyps

    def extract_oracle(self, batched_graph, src_txt):
        graphs = dgl.unbatch(batched_graph)
        hyps = []
        for i, g in enumerate(graphs):
            sent_nodes = g.filter_nodes(get_sent_node_filter_fn())
            labels = g.nodes[sent_nodes].data["labels"]

            hyps.append("\n".join([src_txt[i][j] for j, t in enumerate(labels) if t == 1]))

        return hyps

    def configure_optimizers(self):
        decay_parameters = self.get_parameter_names(self)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
        # optimizer = SWA(optimizer)

        lr_scheduler = self.get_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "learning_rate",
                "frequency": 1,
            },
        }

    def get_parameter_names(self, model, forbidden_layer_types=[nn.LayerNorm]):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())

        return result

    def get_lr_scheduler(self, optimizer):
        def lr_lambda(current_step):
            # current_step += 1
            # return min(
            #     (current_step ** -0.5),
            #     current_step * (self.warmup_steps ** -1.5),
            # )
            return 1

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

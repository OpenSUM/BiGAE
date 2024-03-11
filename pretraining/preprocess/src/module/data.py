from functools import partial

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pyarrow.json import read_json

from src.utils import get_ori_path, get_word_node_filter_fn, get_sent_node_filter_fn


class GraphSumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_fp: str,
        val_fp: str,
        test_fp: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        super(GraphSumDataModule, self).__init__(*args, **kwargs)
        self.train_fp = train_fp
        self.val_fp = val_fp
        self.test_fp = test_fp

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

    @classmethod
    def from_hydra_args(cls, cfg):
        return cls(
            train_fp=cfg.train_file,
            val_fp=cfg.val_file,
            test_fp=cfg.test_file,
            train_batch_size=cfg.train_batch_size,
            val_batch_size=cfg.val_batch_size,
            test_batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
        )

    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset = GraphSumDataset(self.train_fp)
            self.val_dataset = GraphSumDataset(self.val_fp)
        if stage in (None, "test"):
            self.test_dataset = GraphSumDataset(self.test_fp)

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, is_train=True),
        )

        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, is_train=False),
        )

        return loader

    def test_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, is_train=False),
        )

        return loader


class GraphSumDataset(Dataset):
    def __init__(self, fp):
        super(GraphSumDataset, self).__init__()
        self.inner_dataset = read_json(get_ori_path(fp))

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ex = self._convert_table_to_dict(index)

        g = self._build_graph(ex)
        src_txt = ex["src_txt"]
        tgt_txt = ex["tgt_txt"]

        return g, src_txt, tgt_txt

    def _convert_table_to_dict(self, index):
        """convert pyarrow.Table to python dict"""
        keys = self.inner_dataset.column_names
        result = {k: self.inner_dataset[k][index].as_py() for k in keys}

        return result

    def _build_graph(self, ex) -> dgl.DGLGraph:
        labels = ex["labels"]
        edges = ex["edges"]
        word_node_ids = ex["word_node_ids"]
        sent_node_ids = ex["sent_node_ids"]
        word_node_centrality = ex["word_node_centrality"]
        sent_node_centrality = ex["sent_node_centrality"]
        semantic_edge = ex["semantic_edge"]
        topology_edge = ex["topology_edge"]

        u, v = map(list, zip(*edges))
        u, v = u + v, v + u

        n_wnodes, n_snodes = len(word_node_ids), len(sent_node_ids)
        n_nodes = len(word_node_ids) + len(sent_node_ids)
        wnodes = [i for i in range(n_wnodes)]
        snodes = [i + n_wnodes for i in range(n_snodes)]

        g = dgl.graph((u, v), num_nodes=n_nodes)

        # add word feature
        g.nodes[wnodes].data["type"] = torch.zeros(n_wnodes).long()
        g.nodes[wnodes].data["id"] = torch.LongTensor(word_node_ids)
        g.nodes[wnodes].data["centrality"] = torch.LongTensor(word_node_centrality)

        # add sentence node feature
        g.nodes[snodes].data["type"] = torch.ones(n_snodes).long()
        g.nodes[snodes].data["ids"] = torch.LongTensor(sent_node_ids)
        g.nodes[snodes].data["centrality"] = torch.LongTensor(sent_node_centrality)
        g.nodes[snodes].data["position"] = torch.arange(n_snodes)
        g.nodes[snodes].data["labels"] = torch.LongTensor(labels)

        # add edge feature
        semantic_edge = torch.LongTensor(semantic_edge).repeat(2)
        g.edata["semantic"] = semantic_edge

        # add edge feature
        
        # topology_edge = torch.LongTensor(topology_edge)[0::2]
        # g.add_edges(u, v, data={"semantic": semantic_edge, "topology": topology_edge})
        # g.add_edges(v, u, data={"semantic": semantic_edge, "topology": topology_edge})

        return g


class Batch:
    def __init__(self, graph, src_txt, tgt_txt):
        self.graph = graph
        self.src_txt = src_txt
        self.tgt_txt = tgt_txt

    def to(self, device):
        self.graph = self.graph.to(device)

    def __len__(self):
        return self.graph.batch_size


def collator(batched_data, is_train=True):
    graphs, src_txt, tgt_txt = map(list, zip(*batched_data))
    graph_len = [len(g.filter_nodes(get_sent_node_filter_fn())) for g in graphs]  

    sorted_index = np.argsort(graph_len, axis=0)[::-1]
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    src_txt = [src_txt[idx] for idx in sorted_index]
    tgt_txt = [tgt_txt[idx] for idx in sorted_index]

    return Batch(
        graph=batched_graph,
        src_txt=None if is_train else src_txt,
        tgt_txt=None if is_train else tgt_txt,
    )

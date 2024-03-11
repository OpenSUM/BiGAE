from math import floor
import pickle

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from src.utils import read_jsonl, write_jsonl, get_logger, get_ori_path


def discretizer():
    log = get_logger("convert")

    log.info("Read training dataset...")
    train_file = "./data/cnndm/cnndm_train.json"
    train_dataset = read_jsonl(get_ori_path(train_file))

    centrality_sample_ids = np.random.choice(len(train_dataset), size=20000)
    X1 = [
        train_dataset[i]["word_node_centrality"] + train_dataset[i]["sent_node_centrality"]
        for i in centrality_sample_ids
    ]
    X1 = np.array(sum(X1, []))
    X2 = [train_dataset[i]["topology_edge"] for i in centrality_sample_ids]
    X2 = np.array(sum(X2, [])).reshape(-1, 1)

    log.info("Fit node centrality model...")
    node_est = KBinsDiscretizer(n_bins=[10, 10, 10, 10], encode="ordinal", strategy="kmeans").fit(X1)

    log.info("Fit edge centrality model...")
    edge_est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="kmeans").fit(X2)

    log.info("Convert training dataset...")
    for ex in train_dataset:
        ex["word_node_centrality"] = node_est.transform(ex["word_node_centrality"]).tolist()
        ex["sent_node_centrality"] = node_est.transform(ex["sent_node_centrality"]).tolist()
        topology_edge = np.array(ex["topology_edge"]).reshape(-1, 1)
        ex["topology_edge"] = edge_est.transform(topology_edge).reshape(-1).tolist()
        ex["semantic_edge"] = [floor(t * 10.0) for t in ex["semantic_edge"]]

    target_train_file = "./data/cnndm/cnndm_train.jsonl"
    write_jsonl(get_ori_path(target_train_file), train_dataset)
    del train_dataset

    log.info("Read testing dataset...")
    test_file = "./data/cnndm/cnndm_test.json"
    test_dataset = read_jsonl(get_ori_path(test_file))

    log.info("Convert testing dataset...")
    for ex in test_dataset:
        ex["word_node_centrality"] = node_est.transform(ex["word_node_centrality"]).tolist()
        ex["sent_node_centrality"] = node_est.transform(ex["sent_node_centrality"]).tolist()
        topology_edge = np.array(ex["topology_edge"]).reshape(-1, 1)
        ex["topology_edge"] = edge_est.transform(topology_edge).reshape(-1).tolist()
        ex["semantic_edge"] = [floor(t * 10.0) for t in ex["semantic_edge"]]

    target_test_file = "./data/cnndm/cnndm_test.jsonl"
    write_jsonl(get_ori_path(target_test_file), test_dataset)
    del test_file

    log.info("Read validating dataset...")
    val_file = "./data/cnndm/cnndm_valid.json"
    val_dataset = read_jsonl(get_ori_path(val_file))

    log.info("Convert validating dataset...")
    for ex in val_dataset:
        ex["word_node_centrality"] = node_est.transform(ex["word_node_centrality"]).tolist()
        ex["sent_node_centrality"] = node_est.transform(ex["sent_node_centrality"]).tolist()
        topology_edge = np.array(ex["topology_edge"]).reshape(-1, 1)
        ex["topology_edge"] = edge_est.transform(topology_edge).reshape(-1).tolist()
        ex["semantic_edge"] = [floor(t * 10.0) for t in ex["semantic_edge"]]

    target_val_file = "./data/cnndm/cnndm_valid.jsonl"
    write_jsonl(get_ori_path(target_val_file), val_dataset)
    del val_dataset

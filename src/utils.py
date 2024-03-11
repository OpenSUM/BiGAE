import os
import logging
import json

from hydra.utils import get_original_cwd
import torch.distributed as dist
from pytorch_lightning.utilities.distributed import rank_zero_only


def get_logger(name):
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_word_node_filter_fn():
    return lambda node: node.data["type"] == 0


def get_sent_node_filter_fn():
    return lambda node: node.data["type"] == 1


def get_node_filter_fn(type_id):
    return lambda node: node.data["type"] == type_id


def get_edge_filter_fn(src_type_id, dst_type_id):
    return lambda edge: (edge.src["type"] == src_type_id) & (edge.dst["type"] == dst_type_id)


def read_jsonl(fp):
    data = []
    with open(fp) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(fp, data):
    with open(fp, "w") as f:
        for d in data:
            print(json.dumps(d), file=f)


def get_ori_path(path):
    """
    get original path, because hydra creates a directory for each run
    and executing your code within that working directory
    https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory
    """
    if path is None:
        return path

    cwd = get_original_cwd()
    abspath = os.path.join(cwd, path)

    return abspath


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size(group):
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def is_master():
    return get_rank() == 0


def all_gather_object(data, group=None):
    """
    wrapped function for pytorch.distributed.all_gather_object
    """
    if group is None:
        group = dist.group.WORLD

    world_size = get_world_size(group)

    if world_size == 1:
        return [data]

    dist.barrier(group)

    result_list = [None for _ in range(world_size)]
    dist.all_gather_object(result_list, data, group)

    return result_list

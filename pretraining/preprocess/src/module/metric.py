import os
import time
import pickle
import shutil
import logging
import traceback
from typing import List, Union

import torch
import torchmetrics
import numpy as np
import pyrouge
from rouge_score import scoring
import rouge

from src.utils import get_logger


class RougeMetricFromPyRouge(torchmetrics.Metric):
    def __init__(
        self,
        max_enc_buffer_size: int = 4096,
        dist_sync_on_step:bool = False,
    ):
        super(RougeMetricFromPyRouge, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.log = get_logger("rouge-from-pyrouge")
        self.max_enc_buffer_size = max_enc_buffer_size

        assert self.max_enc_buffer_size < 255 * 256, f"unsupported buffer size {self.max_enc_buffer_size}"

        self.add_state("hypothesis", default=[], dist_reduce_fx="cat")
        self.add_state("references", default=[], dist_reduce_fx="cat")

    def update(
        self,
        hyps: Union[str, List[str]],
        refer: Union[str, List[str]],
    ):
        if isinstance(hyps, str):
            hyps = [hyps]

        if isinstance(refer, str):
            refer = [refer]

        pt_buffer_of_hypothesis = torch.stack(tuple(map(self._str_to_pt_buffer, hyps)), 0)
        pt_buffer_of_references = torch.stack(tuple(map(self._str_to_pt_buffer, refer)), 0)

        self.hypothesis.append(pt_buffer_of_hypothesis)
        self.references.append(pt_buffer_of_references)

    def compute(self):
        hypothesis = tuple(map(self._pt_buffer_to_str, self.hypothesis))
        references = tuple(map(self._pt_buffer_to_str, self.references))

        now_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        tmp_dir = f"./rouge_tmp_f{now_time}"

        os.makedirs(os.path.join(tmp_dir, "hypothesis"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "references"), exist_ok=True)

        try:
            for idx, (hyp, ref) in enumerate(zip(hypothesis, references)):
                if len(ref) < 1:
                    continue
                with open(os.path.join(tmp_dir, "hypothesis", f"hyp.{idx}.txt"), "w") as f:
                    f.write(hyp)
                with open(os.path.join(tmp_dir, "references", f"ref.{idx}.txt"), "w") as f:
                    f.write(ref)

            r = pyrouge.Rouge155(log_level=logging.ERROR)
            r.model_dir = os.path.join(tmp_dir, "references")
            r.system_dir = os.path.join(tmp_dir, "hypothesis")
            r.model_filename_pattern = "ref.#ID#.txt"
            r.system_filename_pattern = r"hyp.(\d+).txt"
            rouge_results = r.convert_and_evaluate()
            dict_scores = r.output_to_dict(rouge_results)

        except Exception as e:
            self.log.error(e)
            self.log.error(traceback.format_exc())

        finally:
            shutil.rmtree(tmp_dir)

        scores = {
            "rouge1": round(dict_scores["rouge_1_f_score"], 4) * 100.0,
            "rouge2": round(dict_scores["rouge_2_f_score"], 4) * 100.0,
            "rougel": round(dict_scores["rouge_l_f_score"], 4) * 100.0,
        }

        return scores

    def _str_to_pt_buffer(self, s: str):
        buffer = torch.empty(self.max_enc_buffer_size, device=self.device)
        buffer = buffer.byte()

        enc = pickle.dumps(s)
        enc_size = len(enc)
        assert enc_size + 2 < self.max_enc_buffer_size, \
            f"encoded data exceeds max_size: {enc_size + 2}"

        buffer[0] = enc_size // 255 
        buffer[1] = enc_size % 255
        buffer[2: enc_size + 2] = buffer.new_tensor(list(enc))

        return buffer

    def _pt_buffer_to_str(self, buffer: torch.Tensor):
        size = (255 * buffer[0].item()) + buffer[1].item()

        bytes_list = bytes(buffer[2: size + 2].tolist())
        s = pickle.loads(bytes_list)
        
        return s

    def __hash__(self):
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults.keys():
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))


class RougeMetricFromRouge(torchmetrics.Metric):
    def __init__(
        self,
        use_stemmer: bool = True,
        dist_sync_on_step:bool = False,
    ):
        super(RougeMetricFromRouge, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.log = get_logger("rouge-from-rouge")
        self.aggregator = RougeBatchAggregator()
        self.rouge = rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            stemming=use_stemmer,
        )

        self.add_state("rouge-1", default=[])
        self.add_state("rouge-2", default=[])
        self.add_state("rouge-l", default=[])

    def update(
        self,
        hyps: Union[str, List[str]],
        refer: Union[str, List[str]],
    ):
        if isinstance(hyps, str):
            hyps = [hyps]

        if isinstance(refer, str):
            refer = [refer]

        rouge_results = self.rouge.get_scores(hyps, refer)
        for key, score in rouge_results.items():
            score = torch.tensor([score["p"], score["r"], score["f"]], device=self.device)
            getattr(self, key).append(score)

    def compute(self):
        scores = {key: getattr(self, key) for key in ["rouge-1", "rouge-2", "rouge-l"]}
        self.aggregator.add_scores(scores)
        rouge_results = self.aggregator.aggregate()

        scores = {
            "rouge1": round(rouge_results["rouge-1"].mid.fmeasure, 4) * 100.0,
            "rouge2": round(rouge_results["rouge-2"].mid.fmeasure, 4) * 100.0,
            "rougel": round(rouge_results["rouge-l"].mid.fmeasure, 4) * 100.0
        }

        return scores

    def __hash__(self):
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults.keys():
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))


class RougeBatchAggregator(scoring.BootstrapAggregator):

    def aggregate(self):
        """
        Override function to wrap the final results in `Score` objects.
        This is due to the scores being replaced with a list of torch tensors.
        """
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            scores = [score.cpu().numpy() for score in scores]
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple((scoring.Score(*percentiles[j, :]) for j in range(3)))
            result[score_type] = scoring.AggregateScore(low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def add_scores(self, scores):
        self._scores = scores

import os
import warnings

import hydra
from omegaconf import DictConfig
from transformers.utils.logging import set_verbosity_error
import pytorch_lightning as pl

from src.model import GraphSum
from src.module.data import GraphSumDataModule
from src.utils import get_logger, get_ori_path


class Runner:
    def __init__(self, cfg: DictConfig):
        self.log = get_logger("run")
        self.cfg = cfg

    @classmethod
    def setup_runner(cls, cfg: DictConfig):
        warnings.filterwarnings("ignore")
        set_verbosity_error()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # only for my environment, you can delete it.
        if os.getenv("SLURM_PROCID") is not None:
            os.environ.pop("SLURM_PROCID")

        return cls(cfg)

    def build_datamodule(self, cfg: DictConfig) -> pl.LightningDataModule:
        self.log.info("Instantiating datamodule...")
        datamodule = GraphSumDataModule.from_hydra_args(cfg)
        return datamodule

    def build_model(self, cfg: DictConfig) -> pl.LightningModule:
        self.log.info("Instantiating model...")
        model = GraphSum.from_hydra_args(cfg)
        return model

    def build_trainer(self, cfg: DictConfig) -> pl.Trainer:
        # loggers for pytorch lightning trainer
        tensorboard = pl.loggers.TensorBoardLogger("./runs", "")
        
        # callbakcs for pytorch lightning trainer
        metric = "val_rouge1"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=metric,
            dirpath=os.path.join("./", "checkpoints"),
            filename="{step}-{" + metric + ":.2f}",
            save_top_k=3,
            mode="max",
        )
        # learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        progress_bar_callback = pl.callbacks.RichProgressBar(leave=True)
        model_summary_callback = pl.callbacks.RichModelSummary()

        # instantiate trainer
        self.log.info("Instantiating trainer...")
        trainer: pl.Trainer = hydra.utils.instantiate(
            cfg,
            logger=tensorboard,
            callbacks=[checkpoint_callback, progress_bar_callback, model_summary_callback],
        )

        return trainer

    def train(self) -> None:
        datamodule = self.build_datamodule(self.cfg.get("data"))
        model = self.build_model(self.cfg.get("model"))
        trainer = self.build_trainer(self.cfg.get("trainer"))

        if self.cfg.resume_from_checkpoint:
            self.cfg.resume_from_checkpoint = get_ori_path(self.cfg.resume_from_checkpoint)
            self.log.info(f"Will resume from: {self.cfg.resume_from_checkpoint}")

        self.log.info("Start training!")
        trainer.fit(model, datamodule=datamodule, ckpt_path=self.cfg.resume_from_checkpoint)

    def test(self) -> None:
        datamodule = self.build_datamodule(self.cfg.get("data"))
        model = self.build_model(self.cfg.get("model"))
        trainer = self.build_trainer(self.cfg.get("trainer"))

        if self.cfg.test_from_checkpoint:
            self.cfg.test_from_checkpoint = get_ori_path(self.cfg.test_from_checkpoint)
            self.log.info(f"Will test from: {self.cfg.test_from_checkpoint}")

        self.log.info("Start testing!")
        trainer.test(model, datamodule=datamodule, ckpt_path=self.cfg.test_from_checkpoint)

    def run(self) -> None:
        if self.cfg.mode == "train":
            self.train()
        elif self.cfg.mode == "test":
            self.test()
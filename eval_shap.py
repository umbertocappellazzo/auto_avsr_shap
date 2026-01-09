#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 10:46:06 2026

@author: umbertocappellazzo
"""

import hydra
import torch

from pytorch_lightning import Trainer
from datamodule.data_module import DataModule
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Set modules and trainer
    if cfg.data.modality in ["audio", "visual"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av_shap import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(num_nodes=1, gpus=1, logger=WandbLogger(name="Auto-AVSR_kernel_2000_clean", project="Llama-AVSR"))
    # Training and testing
    modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()

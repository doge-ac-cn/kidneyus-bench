#!/usr/bin/env python3
"""Custom nnU-Net v2 trainers that pin the global seed (plan §5: 3 training
seeds 42/43/44 per trainable model). Everything else is stock nnUNetTrainer
(nnunetv2 2.6.x signature: (plans, configuration, fold, dataset_json, device)).

Usage:
  python scripts/nnunet_trainers.py   # self-check
  nnUNetv2_train Dataset001_KidneyUS 2d 0 -tr nnUNetTrainerSeed42
"""
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _SeedTrainerBase(nnUNetTrainer):
    _seed: int = 42

    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)


class nnUNetTrainerSeed42(_SeedTrainerBase):
    _seed = 42


class nnUNetTrainerSeed43(_SeedTrainerBase):
    _seed = 43


class nnUNetTrainerSeed44(_SeedTrainerBase):
    _seed = 44


if __name__ == "__main__":
    print("trainers importable:",
          [c.__name__ for c in (nnUNetTrainerSeed42, nnUNetTrainerSeed43, nnUNetTrainerSeed44)])

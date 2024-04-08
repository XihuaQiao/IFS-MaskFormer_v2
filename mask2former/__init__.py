# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

from .data import (
    build_detection_test_loader,
    build_detection_train_loader,
    dataset_sample_per_class,
)


from .evaluation.incremental_few_shot_sem_seg_evaluation import IncrementalFewShotSemSegEvaluator

# models

from .MaskFormer import MaskFormer


from .test_time_augmentation import SemanticSegmentorWithTTA

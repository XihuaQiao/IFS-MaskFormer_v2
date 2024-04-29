# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os

import numpy as np
import torch
from torch.nn import functional as F
import cv2

from detectron2.config import configurable
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from mask2former.data.utils import shot_generate

__all__ = ["IncrementalFewShotSemanticDatasetMapper"]


class IncrementalFewShotSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        step=0,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        predictor,
        extra,
        base_classes,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.step = step
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.predictor = predictor

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        self.pseudo_labels = None
        self.pseudo_labels = {}

        self.extra = extra
        self.base_classes = base_classes

    @classmethod
    def from_config(cls, cfg, is_train=True, predictor=None):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
        
        else:
            # TODO decide which way to inference test
            min_size = cfg.INPUT.MIN_SIZE_TEST
            augs = [T.Resize(min_size)]
            dataset_names = cfg.DATASETS.TEST           

        print(dataset_names)
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        _, base_classes, _ = shot_generate(cfg.DATASETS.TASK, cfg.DATASETS.NAME)

        ret = {
            "is_train": is_train,
            "step": cfg.DATASETS.STEP,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "predictor": predictor,
            "extra": cfg.DATASETS.EXTRA,
            "base_classes": base_classes,
        }
        return ret
    
    def save_pseudo_label(self, dataset):
        if not self.predictor:
            return
        for data in dataset:
            image = utils.read_image(data["file_name"], format=self.img_format)
            sem_seg_pred = self.predictor(image)["sem_seg"][0].argmax(dim=0)
            # print(f"make_pseudo_label: {sem_seg_pred.shape} - {sem_seg_pred.unique()}")
            self.pseudo_labels[os.path.basename(data["file_name"])] = sem_seg_pred.cpu().numpy()
            
        del self.predictor

    def gen_pseudo_label(self, file_name, sem_seg_gt, novel_cls):
        sem_seg_pred = self.pseudo_labels[file_name].astype("double")
        base_classes = list(np.unique(sem_seg_pred))
        #TODO base_classes 是否包含0
        if 0 in base_classes:
            base_classes.remove(0)
        for base_cls in base_classes:
            base_and_novel = np.sum((sem_seg_pred == base_cls) & (sem_seg_gt == novel_cls))
            base = np.sum(sem_seg_pred == base_cls)
            novel = np.sum(sem_seg_gt == novel_cls)
            if base_and_novel / base > 0.5 and base_and_novel / novel > 0.5:
                # 抹除这个base class的存在
                # print(f"removing sem_seg_pred of class {base_cls} in image {file_name}")
                sem_seg_pred[sem_seg_pred == base_cls] = 0
        sem_seg_pred[sem_seg_gt == novel_cls] = novel_cls

        return sem_seg_pred


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        edge = None
        if self.extra and self.step == 0:
            sem_seg_gt[sum(sem_seg_gt == i for i in self.base_classes) == 0] = 0
            id = dataset_dict['file_name'].split('/')[-1].split('.')[0]
            edge = cv2.imread(f'/media/data/qxh/workspace/EDTER/VOC_result/png/{id}.png')

        if self.step == 1 and self.is_train:
            novel_cls = dataset_dict["novel_class"]
            if self.pseudo_labels:
                sem_seg_gt = self.gen_pseudo_label(os.path.basename(dataset_dict["file_name"]), sem_seg_gt, novel_cls)
            else:
                sem_seg_gt[sem_seg_gt != novel_cls] = 0

        if edge is not None:
            image, sem_seg_gt, edge = self.aug(image, sem_seg_gt, edge)
            dataset_dict["edge"] = edge.numpy()
        else:
            image, sem_seg_gt = self.aug(image, sem_seg_gt)

        # visualize
        if False:
            if not os.path.exists(f'con_loss/{id}'):
                os.mkdir(f"./con_loss/{id}")
            _img = cv2.cvtColor(np.transpose(image.numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            id = dataset_dict['file_name'].split('/')[-1].split('.')[0]
            label = sem_seg_gt.clone().detach()
            label[label == 0] = 255
            cv2.imwrite(f"./con_loss/{id}/img.jpg", _img)
            cv2.imwrite(f"./con_loss/{id}/baseLabel.png", label.numpy())
            cv2.imwrite(f"./con_loss/{id}/detected_edge.png", edge.numpy())

        # if self.step == 1 and self.is_train:
        #     _img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     name = dataset_dict['file_name'].split('/')[-1].split('.')[0]
        #     cv2.imwrite(f"./pseudo_labels/{name}.jpg", _img)
        #     cv2.imwrite(f"./pseudo_labels/{name}-label.png", sem_seg_gt)
        #     cv2.imwrite(f"./pseudo_labels/{name}-pseudo.png", self.pseudo_labels[os.path.basename(dataset_dict["file_name"])].astype("double"))
        #     cv2.imwrite(f"./pseudo_labels/{name}-gt.png", _gt)

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            dataset_dict["instances"] = self.instance(sem_seg_gt, image_shape)

        return dataset_dict
    
    def aug(self, image, sem_seg_gt, edge=None):
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)

        if edge is not None:
            aug_edge, _ = T.apply_augmentations(transforms, edge)
            edge = aug_edge # (H,W,C)
            edge = torch.as_tensor(edge[:, :, 0].astype("long"))    # (H,W)

        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()

            if edge is not None:
                edge = F.pad(edge, padding_size, value=255).contiguous()

            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        if edge is not None:
            return image, sem_seg_gt, edge
        else:
            return image, sem_seg_gt
    
    def instance(self, sem_seg_gt, image_shape):
        sem_seg_gt = sem_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = np.unique(sem_seg_gt)
        # remove ignored region
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        for class_id in classes:
            masks.append(sem_seg_gt == class_id)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor

        return instances
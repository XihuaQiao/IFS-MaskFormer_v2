# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import numpy as np

import torch, os, cv2, math
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec

from mask2former.data.utils import shot_generate

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.transformer_decoder.position_encoding import PositionEmbeddingSine



@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        novelCriterion: nn.Module,
        baseCriterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pre_norm: bool,
        conv_dim:int,

        step: int,
        base_cls: list,
        total_queries: int,
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.novelCriterion = novelCriterion
        self.baseCriterion = baseCriterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.step = step
        self.base_cls = base_cls[1:]

        # self.query_updates = {k:0 for k in range(total_queries)}
        # self.base_query_updates = {k:0 for k in range(num_queries, total_queries)}
        # self.novel_query_updates = {k:0 for k in range(num_queries)}
        self.query_updates = np.zeros((21, total_queries))
        self.novel_query_updates = np.zeros((21, num_queries))
        self.base_query_updates = np.zeros((21, total_queries - num_queries))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                    #    "loss_con": 0.1,
                    #    "loss_con_base": 0.2,
                    #    "loss_con_query": con_query_weight,
                       }
        
        if cfg.MODEL.KD.ENABLE:
            weight_dict["loss_kd_feature"] = cfg.MODEL.KD.FEATURE_WEIGHT
            weight_dict["loss_kd_class"] = cfg.MODEL.KD.CLASS_WEIGHT
            weight_dict["loss_kd_mask"] = cfg.MODEL.KD.MASK_WEIGHT
            # weight_dict["loss_kd_query"] = 1
            # weight_dict["loss_kd_backbone"] = 0.5

        if cfg.MODEL.CON.ENABLE:
            weight_dict["loss_con_pixel"] = cfg.MODEL.CON.PIXEL_WEIGHT

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["masks", "labels"]
        if cfg.MODEL.KD.ENABLE and cfg.DATASETS.STEP > 0:
            losses.append("kd")
        if cfg.MODEL.CON.ENABLE and cfg.DATASETS.STEP > 0:
            losses.append("con")

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            temperature=cfg.MODEL.KD.TEMPERATURE,
            extra=cfg.DATASETS.EXTRA,
        )

        novelCriterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            # losses=losses.__add__(["con"]) if cfg.MODEL.CON.ENABLE else losses,
            losses=["masks", "labels"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            temperature=cfg.MODEL.KD.TEMPERATURE,
        ) if cfg.DATASETS.STEP > 0 else None
        # ) if cfg.DATASETS.STEP > 0 and cfg.MODEL.MASK_FORMER.NUM_EXTRA_QUERIES > 0 else None

        baseCriterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            # weight_dict={k:v if k != 'loss_kd_class' else 0. for k,v in weight_dict.items()},
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            # losses=losses.__add__(["kd"]) if cfg.MODEL.KD.ENABLE else losses,
            losses=["masks"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            temperature=cfg.MODEL.KD.TEMPERATURE,
        ) if cfg.DATASETS.STEP > 0 else None

        baseCriterion = None
        novelCriterion = None

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "novelCriterion": novelCriterion,
            "baseCriterion": baseCriterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # few shot
            "pre_norm": cfg.MODEL.MASK_FORMER.PRE_NORM,
            "conv_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "step": cfg.DATASETS.STEP,
            "base_cls": shot_generate(cfg.DATASETS.TASK, cfg.DATASETS.NAME)[1],
            "total_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES + cfg.MODEL.MASK_FORMER.NUM_EXTRA_QUERIES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def init_novel_stage(self):
        self.sem_seg_head.predictor.init_novel_stage()

    def seperate(self, outputs, targets):

        #TODO background类别要如何处理？background不参与

        novel_outputs = {k: v[:, :self.num_queries:, ...] for k, v in outputs.items() if k != 'aux_outputs' and k != 'multi_scale_features'}
        base_outputs = {k: v[:, self.num_queries:, ...] for k, v in outputs.items() if k != 'aux_outputs' and k != 'multi_scale_features'}
        # base_outputs["multi_scale_features"] = outputs["multi_scale_features"]

        novel_targets = []
        for tgt in targets:
            _cls = tgt['labels'][[i for i, c in enumerate(tgt['labels']) if c not in self.base_cls and c != 0]]
            _mask = tgt['masks'][[i for i, c in enumerate(tgt['labels']) if c not in self.base_cls and c != 0], ...]
            novel_targets.append({
                'labels': _cls,
                'masks': _mask,
            })

        base_targets = []
        for tgt in targets:
            _cls = tgt['labels'][[i for i, c in enumerate(tgt['labels']) if c in self.base_cls]]
            _mask = tgt['masks'][[i for i, c in enumerate(tgt['labels']) if c in self.base_cls], ...]
            base_targets.append({
                'labels': _cls,
                'masks': _mask,
            })            

        return novel_outputs, novel_targets, base_outputs, base_targets


    def forward(self, batched_inputs, outputs_old=None):

        # print(batched_inputs[0]['file_name'])

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        # outputs["backbone"] = features

        if self.training:
            assert "instances" in batched_inputs[0], f"There are no instances in batch_inputs! ({batched_inputs[0].keys()})"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)

            losses = {}
            losses = self.criterion(outputs, targets, outputs_old,
                                    [x['image'].shape for x in batched_inputs],
                                    [x['file_name'] for x in batched_inputs],
                                    [x['edge'] for x in batched_inputs] if 'edge' in batched_inputs[0].keys() else None
            )

            # print("criterion: ", end='')
            for indice in self.criterion.indices:
                for (qIdxs, clses) in indice:
                    for qIdx, cls in zip(qIdxs, clses):
                        self.query_updates[cls, qIdx] += 1
            #             print(f"{qIdx} - {cls}, ", end='')
            # print("")

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            
            if self.novelCriterion or self.baseCriterion:
                novel_outputs, novel_targets, base_outputs, base_targets = self.seperate(outputs, targets)

            if self.novelCriterion:
                # print("novel criterion: ", end='')
                novel_losses = self.novelCriterion(novel_outputs, novel_targets, outputs_old)
                for indice in self.novelCriterion.indices:
                    for (qIdxs, clses) in indice:
                        for qIdx, cls in zip(qIdxs, clses):
                            self.novel_query_updates[cls, qIdx] += 1
                #             print(f"{qIdx} - {cls}, ", end='')
                # print("")
                
                for k in list(novel_losses.keys()):
                    if k in self.novelCriterion.weight_dict:
                        losses[f"novel_{k}"] = self.novelCriterion.weight_dict[k] * novel_losses[k] * 0.5

            if self.baseCriterion:
                base_losses = self.baseCriterion(base_outputs, base_targets, outputs_old)
                # print("base criterion: ", end='')
                for indice in self.baseCriterion.indices:
                    for (qIdxs, clses) in indice:
                        for qIdx, cls in zip(qIdxs, clses):
                            self.base_query_updates[cls, qIdx] += 1
                #             print(f"{qIdx} - {cls}, ", end='')
                # print("")

                for k in list(base_losses.keys()):
                    if k in self.baseCriterion.weight_dict:
                        losses[f"base_{k}"] = self.baseCriterion.weight_dict[k] * base_losses[k] * 0.02

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []

            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)    
                processed_results.append(r)

            outputs["sem_seg"] = processed_results

            return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

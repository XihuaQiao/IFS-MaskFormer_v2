# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import copy
import logging
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
import random

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from mask2former.modeling.contrastive_learning.utils import *

from mask2former.data.datasets.register_voc_fewshotseg import CLASS_NAMES

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def knowledge_distillation_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        temperature: float,
        dim: int,
):
    # targets: (bn, Q, n_cls)

    # inputs = inputs.flatten(2, -1)
    # targets = targets.flatten(2, -1)
    # print(f"KD_LOSS: targets: {targets.shape}, inputs: {inputs.shape}")
    inputs = torch.log_softmax(inputs[:, :targets.shape[1], ...] / temperature, dim=dim)
    targets = torch.softmax(targets / temperature, dim=dim)

    loss = F.kl_div(inputs, targets, reduction=reduction) * temperature * temperature
    # print(f"criterion: loss - {loss.shape}")

    return loss

'''
# https://github.com/fcdl94/MiB/blob/master/utils/loss.py
def knowledge_distillation_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        alpha: float,
):
    inputs = inputs.narrow(1, 0, targets.shape[1])

    outputs = torch.log_softmax(inputs, dim=-1)
    labels = torch.softmax(targets * alpha, dim=-1)

    loss = (outputs * labels).mean(dim=-1)

    if reduction == 'mean':
        outputs = -torch.mean(loss)
    elif reduction == 'sum':
        outputs = -torch.sum(loss)
    else:
        outputs = -loss

    return outputs
'''
    
knowledge_distillation_loss_jit = torch.jit.script(
    knowledge_distillation_loss
)

def contrastive_loss(
        input1: torch.Tensor,
        input2: torch.Tensor,
):
    ## 欧氏距离
    # pdist = nn.PairwiseDistance(p=2)
    # res = pdist(input1, input2)
    # # use batch mean
    # return  - res.sum() / res.shape[0]    

    # 余弦相似度
    res = F.cosine_similarity(input1, input2, dim=-1)

    return res.sum() / res.shape[0]

contrastive_loss_jit = torch.jit.script(
    contrastive_loss
)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, temperature, extra=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.temperature = temperature
        self.extra = extra

        self.strong_queries = [7, 10, 25, 30, 38, 39, 43, 65, 66, 75, 77, 79, 92, 99]

        self.L2_loss = nn.MSELoss()

    def loss_labels(self, outputs, targets, indices, num_masks, outputs_old):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        # if outputs["pred_logits"].shape[1] > 100:
        # indices = [indices[0]]
        
        for indice in indices:
            idx = self._get_src_permutation_idx(indice)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indice)])
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, outputs_old):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        losses = {"loss_mask": torch.tensor(0.0).to(outputs["pred_masks"].device),
                  "loss_dice": torch.tensor(0.0).to(outputs["pred_masks"].device)}
        
        # indices = [indices[0]]

        for i, indice in enumerate(indices):

            src_idx = self._get_src_permutation_idx(indice)
            tgt_idx = self._get_tgt_permutation_idx(indice)

            src_masks = outputs["pred_masks"]
            src_masks = src_masks[src_idx]
            masks = [t["masks"] for t in targets]
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)
            target_masks = target_masks[tgt_idx]

            # No need to upsample predictions as we are using normalized coordinates :)
            # N x 1 x H x W
            src_masks = src_masks[:, None]
            target_masks = target_masks[:, None]

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks, 
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            if i > 0:
                w = 0.5
            else:
                w = 1

            losses["loss_mask"] += sigmoid_ce_loss_jit(point_logits, point_labels, num_masks) * w
            losses["loss_dice"] += dice_loss_jit(point_logits, point_labels, num_masks) * w

        del src_masks
        del target_masks
        return losses
    
    def loss_kd(self, outputs, targets, indices, num_masks, outputs_old):
        assert outputs_old is not None, "Using loss_kd should provide outputs_old!"
        # print(f"outputs - {outputs['query_features'].shape}, outputs_old - {outputs_old['query_features'].shape}")
        if outputs['pred_logits'].shape[1] == len(self.strong_queries):
            outputs_old['pred_logits'] = outputs_old['pred_logits'][:, self.strong_queries, ...]
            outputs_old['pred_masks'] = outputs_old['pred_masks'][:, self.strong_queries, ...]

        losses  = {
            "loss_kd_mask": self.L2_loss(outputs["pred_masks"][:, :outputs_old['pred_masks'].shape[1], ...], outputs_old["pred_masks"]),     # (B,Q,H,W)                                                 
            "loss_kd_class": knowledge_distillation_loss_jit(
                outputs["pred_logits"][:, :outputs_old['pred_logits'].shape[1], :-1], outputs_old["pred_logits"][:, :, :-1], 'batchmean', self.temperature, -1), # (B,Q,C+1)    
        }

        if 'multi_scale_features' in outputs.keys():
            idx = [0]
            feat = [outputs["multi_scale_features"][i] for i in idx]
            feat_old = [outputs_old["multi_scale_features"][i] for i in idx]
            losses['loss_kd_feature'] = sum([self.L2_loss(output, output_old) 
                                             for output, output_old in zip(feat, feat_old)]) / len(idx)
        
        if 'backbone_features' in outputs.keys():
            idx = ['res5']
            feat = [outputs["backbone_features"][i] for i in idx]
            feat_old = [outputs_old["backbone_features"][i] for i in idx]
            losses['loss_kd_backbone'] = sum([self.L2_loss(output, output_old) 
                                             for output, output_old in zip(feat, feat_old)]) / len(idx)     # （B,Q,D)

        return losses
    
    def loss_con(self, outputs, targets, indices, num_masks, outputs_old):
        feature = outputs["multi_scale_features"][-1]   # B,D,H,W 
        '''
        segments = outputs["pred_masks"].sigmoid()      # B,Q,H,W
        clses = F.softmax(outputs['pred_logits'], dim=-1)[..., :-1]

        # TODO 这样激活是否合理
        segments[segments >= 0.5] = 1.0
        segments[segments < 0.5] = 0.0
        # segments = segments.int()

        loss_con = torch.tensor(0.0).to(feature.device)
        batch = feature.shape[0]

        batch_labels = []
        excluded_pixels = []
        for tgt in targets:
            idx = 0 if tgt['labels'][0] !=0 else 1
            tmp = np.zeros_like(tgt['masks'][0].cpu().numpy())
            batch_labels.append(tgt['masks'][idx:, ...])

            for lbl in tgt['masks'][idx:, ...]:
                tmp[lbl.cpu().numpy() == 1] = 1

            excluded_pixels.append(tmp)
        
        batch_labels = torch.cat(batch_labels, dim=0)
        batch_labels = [batch_labels[i] for i in range(batch_labels.shape[0])]      # neg_segments

        for idx in range(batch):     # batch
            name = self.names[idx].split('/')[-1].split('.')[0]
            if not os.path.exists(f'con_loss/{name}'):
                os.mkdir(f"./con_loss/{name}")
            # visualize the result of mask2former query-mask
            if True:
                masks = outputs["pred_masks"][idx, ...].sigmoid()     # Q,H,W
                # upsample masks
                masks = F.interpolate(
                    masks.unsqueeze(0),
                    size=(self.img_shapes[idx][-2], self.img_shapes[idx][-1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().detach().cpu().numpy()
                masks_T = masks.copy()
                threshold = 0.5
                masks_T[masks_T > threshold] = 1
                masks_T[masks_T <= threshold] = 0

                # masks_T_with_label = masks_T        # mask包含base class
                # masks_T[:, excluded_pixels[idx] == 1] = 0   # mask不包含base label
                # masks = torch.einsum('qhw,chw->qchw', masks, self.images[idx]).detach()
                # 只考虑除了base label以外的pixel
                confidences = []
                for i in range(masks.shape[0]):
                    mask = masks[i, ...]
                    _mask_T = masks_T[i, ...]
                    cls = torch.argmax(clses[idx, i, ...])
                    # 检测边缘时，将base label都加入，让每个mask都具有base label的边缘，从而确保公平性
                    # confidence只计算base label以外部分，但会不可避免地计算到一些base label的边缘pixel
                    if cls == 0:
                        masks_T[i, ...][excluded_pixels[idx] == 1] = 0
                        _mask_T[excluded_pixels[idx] == 1] = 1
                        _mask = mask[_mask_T == 0]
                        confidence = (1 - _mask[_mask <= threshold]).sum() / (_mask_T == 0).sum()
                    else:
                        masks_T[i, ...][excluded_pixels[idx] == 1] = 1
                        _mask_T[excluded_pixels[idx] == 1] = 0
                        _mask = mask[_mask_T == 1]
                        confidence = _mask[_mask > threshold].sum() / _mask_T.sum()
                    # if cls == 0:    # background
                    #     mask_T[excluded_pixels[idx] == 1] = 0
                    # else:
                    #     mask_T[excluded_pixels[idx] == 1] = 1
                    # _mask_T[excluded_pixels[idx] == 1] = 0

                    # _mask = mask[_mask_T == 1]
                    # confidence = _mask[_mask > threshold].sum() / _mask_T.sum()

                    plt.imsave(f"./con_loss/{name}/query_{i:02d}_{CLASS_NAMES[cls]}.png", mask, cmap='gray')
                    plt.imsave(f"./con_loss/{name}/query_{i:02d}_{CLASS_NAMES[cls]}_T_{confidence}.png", _mask_T, cmap='gray')
                    confidences.append(confidence)

                selecting_segments(masks_T, name, self.det_edges[idx], confidences, kernel_size=3)

            # with torch.no_grad():
            #     selected = scoring_segments(feature[idx, ...], segments[idx, ...], threshold=0.5)
            # if len(selected) == 0:
            #     continue

            # # visualize the result of socring_segments
            # masks = torch.stack(selected, dim=0)
            # if torch.sum(masks) > 0:
            #     masks = F.interpolate(
            #         masks.unsqueeze(0).float(),
            #         size=(self.images[idx].shape[-2], self.images[idx].shape[-1]),
            #         mode='bilinear',
            #         align_corners=False
            #     )
            #     masks = torch.sum(masks.squeeze(0), dim=0)
            #     masks[masks >= 1] = 1
            #     masks[masks == 0] = 0.1
            #     img = torch.einsum('chw,hw->chw', self.images[idx], masks).detach()

            #     img = cv2.cvtColor(np.transpose(img.cpu().numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(f"./con_loss/{name}/img_mask.jpg", img)

            # _loss = InfoNCE_cotrastive_loss(feature[idx, ...], selected, batch_labels, excluded_pixels[idx], temperature=0.1, reduction='mean')
            # loss += _loss

        ### 训练过程中加入基于ground truth的对比学习
        '''        
        return {
            # "loss_con": loss_con,
            "loss_con_pixel": loss_con_base(targets, feature, temperature=0.1),
            }
    
    # Try——1 使用query对应的mask组成初始mask，然后一步一步合并，生成最后的无监督学习mask，暂时先不使用文章中提到的聚类思想

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, outputs_old):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'kd': self.loss_kd,
            'con': self.loss_con,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, outputs_old)

    def forward(self, outputs, targets, outputs_old, img_shapes=None, names=None, det_edges=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        self.img_shapes = img_shapes
        self.names = names
        self.det_edges = det_edges
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # for analyze the model 
        self.indices = copy.deepcopy(indices)
        for i in range(len(self.indices)):
            for j in range(len(self.indices[i])):
                (_, J) = self.indices[i][j]
                clses = targets[j]["labels"]
                self.indices[i][j] = (_.tolist(), [clses[j].item() for j in J])
        # for i in range(len(self.indices)):
        #     # print(self.indices[i])
        #     (_, J) = self.indices[i]
        #     clses = targets[i]["labels"]
        #     self.indices[i] = (_.tolist(), [clses[j].item() for j in J])
        #     # print(f"query {self.indices[i][0]} - class {[CLASS_NAMES[idx] for idx in self.indices[i][1]]}")

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, outputs_old))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and outputs["aux_outputs"] is not None and not self.extra:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in ['masks', 'labels']:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, None)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    

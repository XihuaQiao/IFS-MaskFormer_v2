import torch
import torch.nn.functional as F

import torch.nn as nn
from abc import ABC

import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional

def max_index(matrix):
    m = -2
    row = 0
    line = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > m and i != j:
                row = i
                line = j
                m = matrix[i, j]

    if row > line:
        return line, row
    else:
        return row, line


def scoring_segments(feature, masks, threshold=0.7):
    '''
        https://github.com/Sreyan88/MMER/blob/2b076fb11dd2f04fcae6800a1fcc7e54a13e4ba5/src/infonce_loss.py
        feature: D,H,W
        masks: N,H,W
        threshold:

        outputs: list [segments] (H,W)
    '''

    #TODO 筛选和gt还有pseud label冲突的segment，目前不进行筛选，只对于重合的pixel置0
    '''
        进行筛选：
            1. 将和gt和
    '''

    # merged_S = []          # 所有被合并过的segment
    S = [masks[i, ...] for i in range(masks.shape[0]) if torch.sum(masks[i, ...]) != 0]          # 所有未被合并的segment
    all_segments = [masks[i, ...] for i in range(masks.shape[0]) if torch.sum(masks[i, ...]) != 0]       # 所有segments，包括合并后的，用来记载score，按照顺序对应
    scores = [0 for i in range(len(S))]             # 用来记载all_segments中的分数
    mapping = [i for i in range(len(S))]          # 记载 S - score的对应, index - idx of score

    prototypes = []
    for segment in S:
        p = torch.einsum('nhw,hw->n', feature, segment.float()) / segment.sum()
        prototypes.append(p)

    m = len(S)
    while(len(S) > 1):
        assert len(prototypes) == len(S), f"scoring segments: len of prototypes {len(prototypes)} != len of Segments {len(S)}"
        p = torch.stack(prototypes, dim=0)
        # compute similarity matrix
        p = p / torch.norm(p, dim=-1, keepdim=True)
        similarity_matrix = torch.mm(p, p.T)

        if len(S) == m:
            matrix = similarity_matrix.detach()
            print(f"similarity matrix: {torch.argmax(torch.sum(matrix, dim=-1))}")

        a, b = max_index(similarity_matrix)
        segment_a = S[a]
        segment_b = S[b]
        # segment_c = segment_a + segment_b
        # segment_c[segment_c > 0] = 1.0
        # merged_S.append(segment_a)
        # merged_S.append(segment_b)

        # a < b
        S.pop(a)
        S.pop(b - 1)
        
        # TODO
        # S.append(segment_c)

        # if s = c, u = 0
        # all_segments.append(segment_c)
        # scores.append(0)

        # update mapping when scores change
        mapping.pop(a)
        mapping.pop(b - 1)
        # mapping.append(len(scores) - 1)
    
        # print(f"scoring_segments: scores - {len(scores)}, S - {len(S)}")
        # otherwise u = u + (1 - a * b)
        
        # for i in range(len(S) - 1):
        for i in range(len(S)):
            scores[mapping[i]] += 1 - (F.cosine_similarity(prototypes[a], prototypes[b], dim=-1)).item()

        prototypes.pop(a)
        prototypes.pop(b - 1)
        # prototypes.append(torch.einsum('nhw,hw->n', feature, segment_c) / segment_c.sum())

    outputs = []
    for i, (segment, score) in enumerate(zip(all_segments, scores)):
        if score > threshold:
            flag = True
            for _s in outputs:
                if ((segment + _s) == 2).any():
                    flag = False
                    break
            if flag:
                outputs.append(segment)
                print(f"segment {i} has been chosen, scoring {score}, {torch.sum(segment)}")

    print(f"total {len(outputs)} segments has been chosen.")

    return outputs


def compute_IoU(pred, target):
    pred_ins = pred == 0
    target_ins = target == 0
    inser = pred_ins[target_ins].sum()
    union = pred_ins.sum() + target_ins.sum() - inser
    iou = inser / union

    return iou


def selecting_segments(segments, filename, detected_edge, confidences, kernel_size=4, confidence_threshold=0.9):
    '''
    基于segment的边缘轮廓与EDTER结果的重合程度筛选maskformer的segment，以便后续的对比学习，该fucntion仅针对一张图片
        segments: segments of one image, (Q,H,W)
        filename: name of input image, to get the result of edge detection
        kernel_size: 调节膨胀的大小，数字越大边缘线条越粗
    '''
    
    ious = []

    for i in range(len(segments)):
        # s = segments[i].detach().cpu().numpy()
        s = np.array(segments[i] * 255, dtype=np.uint8)
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)     # (H,W,3)

        edge = cv2.Laplacian(s, cv2.CV_64F)
        edge = cv2.convertScaleAbs(edge)            # (H,W,3)
        # edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)   # (H,W) 此时segment的edge是白线，即1为边缘
        assert set(np.unique(edge).tolist()) <= set([0, 255]), f"{id} query - {i}, edge contains {np.unique(edge)}"
        # 对调edge中的0和255，让白线变成黑线
        _edge = np.zeros_like(edge)
        _edge[edge == 0] = 255
        edge = _edge[:, :, 0]

        # 对于边缘结果，0是边缘，1是背景，因此黑线为边缘结果，对于EDTER的结果和segment的边缘都适用
        # detected_edge = cv2.imread(f'/media/data/qxh/workspace/EDTER/VOC_result/png/{filename}.png', cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_edge = cv2.erode(edge, kernel)                     # (H,W,3)

        detected_edge[detected_edge >= 128] = 255
        detected_edge[detected_edge < 128] = 0
        dilation_detected_edge = cv2.erode(detected_edge.astype('uint8'), kernel)   # (H,W)

        iou = compute_IoU(dilation_edge, dilation_detected_edge)
        ious.append(iou)

    l = []
    for idx in np.argsort(-np.array(ious)):
        if len(l) == 5:
            break
        # TODO confidence阈值设定很重要，这个confidence的定义是否有理
        if confidences[idx] > confidence_threshold:
            l.append(idx)

    print({filename})
    for idx in l:
        print(f"query {idx} - {ious[idx]}")

    return ious

def InfoNCE_cotrastive_loss(
        feature: torch.Tensor,
        pos_segments: list,
        neg_segments: list,
        excluded_pixels: torch.Tensor,
        temperature: float,
        # negative_mode: str,
        reduction: str,
):
    '''
    https://github.com/Sreyan88/MMER/blob/2b076fb11dd2f04fcae6800a1fcc7e54a13e4ba5/src/infonce_loss.py
    loss of one image within batch size negative segments
        feature: feature of one image (D, H, W)
        pos_segments: list of potential segments in current image N*(H, W)
        neg_segments: list of pseudo label or ground truth within this batch M*(H, W)
        exluded_pixels: pixels of ground truth & pseudo label, (H, W)
    '''

    # TODO 目前使用的策略，是对于segment的所有pixel都进行相对于该segment prototyp以及其他negative segment的对比学习，和论文内方法稍有不同

    # print(f"InfoNCE_loss: positive segment - {pos_sterminaegments[0].shape}, excluded_pixels - {excluded_pixels.shape}, negative segment - {neg_segments[0].shape}")
    # pos_segments = [torch.einsum('hw,hw->hw', s, 1 - excluded_pixels) for s in pos_segments]

    # TODO 目前将label进行下采样，interpolate会对区域进行平均然后下采样，即0.5，0.75这样
    with torch.no_grad():
        for i in range(len(neg_segments)):
            s = neg_segments[i]
            s = F.interpolate(
                s.unsqueeze(0).unsqueeze(0).float(), 
                size=(feature.shape[-2], feature.shape[-1]), 
                mode='bilinear',
                align_corners=False
            ).squeeze()
            s[s > 0.5] = 1.0
            s[s < 0.5] = 0.0
            s[s == 0.5] = random.choice([0.0, 1.0])
            neg_segments[i] = s

    pos_prototypes = [torch.einsum('dhw,hw->d', feature, s) / torch.sum(s) for s in pos_segments]
    neg_prototypes = [torch.einsum('dhw,hw->d', feature, s) / torch.sum(s) for s in neg_segments]
    neg_prototypes = torch.stack(neg_prototypes, dim=0)     # (M,D)

    pos_prototypes = [F.normalize(p, dim=-1) for p in pos_prototypes]
    neg_prototypes = F.normalize(neg_prototypes, dim=-1)

    # feature = torch.einsum('dhw,hw->dhw', feature, 1 - excluded_pixels)    

    loss = 0

    for pos_segment, pos_prototype in zip(pos_segments, pos_prototypes):
        query = torch.einsum('dhw,hw->dhw', feature, pos_segment)
        query = query.transpose(-3, -1).reshape(-1, pos_prototype.shape[0])     # (h*w, d)
        pos_logit = torch.sum(torch.einsum('nd,d->nd', query, pos_prototype), dim=1, keepdim=True)  # (h*w, 1)
    
        # negative_mode = unpaired, each pos_segment pair all negative segment
        neg_logits = torch.einsum('nd,md->nm', query, neg_prototypes)           # (h*w, m)

        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        loss += F.cross_entropy(logits / temperature, labels, reduction=reduction)

    return loss


def loss_con_base(targets, features, temperature=0.1):
    batch_info = []

    labels = []
    prototypes = []
    backgrounds = []

    # features上采样到512 * 512
    features = F.interpolate(
        features,
        size=(targets[0]['masks'].shape[-2], targets[0]['masks'].shape[-1]),
        mode='bilinear',
        align_corners=False,
    )

    #TODO background 是否要参与到对比学习中, 目前不参与
    useBackground = False
    for batch, tgt in enumerate(targets):
        for i in range(len(tgt['labels'])):
            label = tgt['labels'][i]
            mask = tgt['masks'][i].float()
            if label in [16, 17, 18, 19, 20]:
                prototype = torch.einsum('dhw,hw->d', features[batch], mask) / torch.sum(mask)
                batch_info.append({
                    'batch': batch,
                    'label': label,
                    'prototype': prototype,
                    'mask': mask,
                })
                labels.append(label)
                prototypes.append(prototype)
        idx = torch.nonzero(torch.eq(tgt['labels'], 0))
        if len(idx) == 1:
            backgrounds.append({
                'batch': batch,
                'prototype': torch.einsum('dhw,hw->d', features[batch], tgt['masks'][i].float()) / tgt['masks'][i].float().sum(),
            })

    loss = torch.tensor(0.0).to(features.device)
    
    prototypes = [F.normalize(p, dim=-1) for p in prototypes]
    
    if len(prototypes) == 0:
        assert len(labels) == 0, f"ERROR: there are {len(labels)} labels except background in current batch!"
        return loss

    labels = torch.Tensor(labels)
    prototypes = torch.stack(prototypes, dim=0)     # (M, d)

    #TODO 目前把自身也作为正样本加入计算
    _mask = torch.eq(labels, labels.reshape(-1, 1)).float().to(features.device)
    # _mask = _mask - torch.eye(len(labels), dtype=torch.float, device=features.device)       # 当label_i == label_j时为1，即正样本矩阵, (M, M)
    logit_mask = torch.ones_like(_mask)
    # logit_mask = torch.ones_like(_mask) - torch.eye(len(labels), dtype=torch.float, device=features.device)          # 对角线为0的矩阵，即自己不与自己进行计算 (M, M)

    for idx, info in enumerate(batch_info):
        label = info['label']
        mask = info['mask']
        batch_idx = info['batch']

        query = torch.einsum('dhw,hw->dhw', features[batch_idx], mask)
        query = query.transpose(-3, -1).reshape(-1, features[batch_idx].shape[0])     # (h*w, d) 
        query = query[torch.sum(query, dim=1) != 0]     # (N, d)

        logits = torch.div(torch.einsum('Nd,Md->NM', query, prototypes), temperature)   # (N, M)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_logits = logits[:, _mask[idx] == 1]         # P个正样本 (N, P)
        all_logits = logits[:, logit_mask[idx] == 1]    # A个除样本自身以外的样本，A == M - 1, (N, M - 1)
        if useBackground and len(backgrounds) > 0:
            bp_prototypes = torch.stack([bp['prototype'] for bp in backgrounds], dim=0) # (B, d)
            bp_logits = torch.div(torch.einsum('Nd,Bd->NB', query, bp_prototypes), temperature)
            #TODO 减去max是否要在最开始一起操作
            bp_logits_max, _ = torch.max(bp_logits, dim=1, keepdim=True)        # (N, B)
            bp_logits = bp_logits - bp_logits_max.detach()
            all_logits = torch.cat([all_logits, bp_logits], dim=1)

        sum_pos_logits = pos_logits.sum(dim=1)        # (N)
        log_all_logits = torch.log(torch.exp(all_logits).sum(dim=1))     # (N)

        log_prob = sum_pos_logits - log_all_logits
        mean_log_prob_pos =  - log_prob / _mask[idx].sum()

        loss += mean_log_prob_pos.mean()

    loss = loss / len(batch_info)

    return loss
 

# https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/loss/loss_contrast.py
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, temperature, base_temperature, ignore_label=[], max_samples=1024, max_views=100, weight=1):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_label

        self.max_samples = max_samples      # 一个batch中所有对比pixel的个数最大值
        self.max_views = max_views          # 一张图片中一个cls的pixel个数最大值

        self.weight = weight

    def _hard_anchor_sampling(self, X, y_hat, y, filenames):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x not in self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]   # cls在图片中的占比要至少大于max_view

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()   # 一个batch中[所有出现类别，每个了类别选中的pixel数]
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                # idx = torch.zeros((128 * 128), dtype=torch.float)
                # idx[hard_indices] = 1
                # idx[easy_indices] = 0.5
                # idx = idx.reshape(128, 128)
                # name = filenames[ii].split('/')[-1].split('.')[0]
                # plt.imsave(f"contrast/{name}_cls{cls_id}.png", idx, cmap='gray')

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits + 1e-10)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, targets, outputs, filenames):
        feats = outputs['embedding']     # 128,128
        batch_size = feats.shape[0]

        labels = torch.zeros((batch_size, targets[0]['masks'].shape[1], targets[0]['masks'].shape[2]), dtype=torch.float).cuda()
        for b in range(batch_size):
            for lbl, mask in zip(targets[b]['labels'], targets[b]['masks']):
                labels[b][mask] = lbl.float()   # 512,512

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        mask_cls = F.softmax(outputs['pred_logits'], dim=-1)[..., :-1]
        mask_pred = outputs['pred_masks'].sigmoid()
        sem_seg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        predict = sem_seg.argmax(dim=1) # B,H,W

        if predict.shape[-1] != feats.shape[-1]:
            predict = predict.unsqueeze(1).float().clone()
            predict = torch.nn.functional.interpolate(predict,
                                                    (feats.shape[2], feats.shape[3]), mode='nearest')
            predict = predict.squeeze(1).long()

        assert predict.shape[-1] == labels.shape[-1] and labels.shape[-1] == feats.shape[-1], f"contrast: \
            predict - {predict.shape}, labels - {labels.shape}, feats - {feats.shape}"

        # for i in range(batch_size):
        #     name = filenames[i].split('/')[-1].split('.')[0]
        #     plt.imsave(f"./contrast/{name}_gt.png", labels[i].cpu().numpy(), cmap='gray')
        #     plt.imsave(f"./contrast/{name}_pred.png", predict[i].cpu().numpy(), cmap='gray')

        labels = labels.contiguous().view(batch_size, -1)       # B,H*W
        predict = predict.contiguous().view(batch_size, -1)     # B,H*W

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # B,D,H,W -> B,H*W,D

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict, filenames)

        loss = self._contrastive(feats_, labels_)
        return loss

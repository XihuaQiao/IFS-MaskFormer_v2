import torch
import torch.nn.functional as F

import random
import cv2
import numpy as np

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
 
# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from mask2former.data.utils import *


CLASS_NAMES = (
    'background',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
)

def get_metadata(task, dataname):
    # actually all the stuff below should be thing!!!

    class_list, base_list, novel_list = shot_generate(task, dataname)
    BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in base_list]
    NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in novel_list]

    meta = {}
    stuff_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_colors.insert(0, [0, 0, 0])
    meta["stuff_classes"] = CLASS_NAMES
    meta["novel_classes"] = NOVEL_CLASS_NAMES
    meta["base_classes"] = BASE_CLASS_NAMES
    meta["stuff_colors"] = stuff_colors

    stuff_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id[0] = 0
    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    return meta


def register_all_coco(task, step, shot, ishot, extra=False):
    root = '/media/data/qxh/workspace/IFS-MaskFormer/data/coco'

    meta = get_metadata(task, 'coco')

    for name, step, filename in [
        ("train", 0, 'train_step0.pth'),
        ("train", 1, 'train_step1.pth'),
        ("val", 0, 'valid.pth')
    ]:
        image_dir = os.path.join(root, name + '2017')
        gt_dir = os.path.join(root, 'annotations', 'coco_masks', 'instance_' + name + '2017')
        d = f'list/coco/{task}/{filename}'

        # print(all_name, 'x'*500)
        if name == "train" and step == 0:
            all_name = f"coco_IFS_sem_seg_{name}_step{step}"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_base_seg(d, root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )
            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="IFS_sem_seg",
                ignore_label=255,
                **meta,
            )
        elif name == "train" and step == 1:
            all_name = f"coco_IFS_sem_seg_{name}_step{step}"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_novel_seg(d, step, task, shot, dataname='coco', root=root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )

            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="IFS_sem_seg",
                ignore_label=255,
                **meta,
            )
        elif name == 'val':
            all_name = f"coco_IFS_sem_seg_val"
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_base_seg(d, root),
                # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
            )
            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="IFS_sem_seg",
                ignore_label=255,
                **meta,
            )

        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="IFS_sem_seg",
            ignore_label=255,
            **meta,
        )

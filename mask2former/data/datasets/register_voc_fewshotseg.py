# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import load_sem_seg
from mask2former.data.utils import *


CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

def _get_voc_meta(task, dataname):

    class_list, base_list, novel_list = shot_generate(task, dataname)
    BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in base_list]
    NOVEL_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if i in novel_list]

    meta = {
        "stuff_classes": CLASS_NAMES,
        "stuff_colors": CLASS_COLORS,
        "novel_classes": NOVEL_CLASS_NAMES,
        "base_classes": BASE_CLASS_NAMES,
    }

    return meta


def register_all_pascal(task, step, shot, ishot, extra=False):
    root = '/media/data/qxh/workspace/IFS-MaskFormer/data/voc'

    meta = _get_voc_meta(task, 'voc')
    image_dir = os.path.join(root, 'JPEGImages')
    gt_dir = os.path.join(root, 'SegmentationClassAug')

    for step in [0, 1]:
        # prepare train dataset
        d = f"list/voc/{task}/train_step{step}{'_extra' if extra else ''}.pth"
        all_name = f"voc_IFS_sem_seg_train_step{step}"
        if step > 0 and not extra:
            all_name = f"{all_name}_shot_{shot}"
        if step == 0 or extra:
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_base_seg(d, root),
            )
        else:
            DatasetCatalog.register(
                all_name,
                lambda d=d: load_novel_seg(d, task, shot, ishot, dataname='voc', root=root),
            )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="IFS_sem_seg",
            ignore_label=255,
            **meta,
        )

    # prepare valid dataset
    d = f'list/voc/{task}/valid.pth'
    all_name = f"voc_IFS_sem_seg_val"
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

    # for name, step, filename in [
    #     ("train", 0, 'train_step0.pth'),
    #     ("train", 1, 'train_step1.pth'),
    #     ("val", 0, 'valid.pth')
    # ]:
    #     image_dir = os.path.join(root, 'JPEGImages')
    #     gt_dir = os.path.join(root, 'SegmentationClassAug')
    #     d = f'list/voc/{task}/{filename}'

    #     if name=="train" and step == 0:
    #         all_name = f"voc_IFS_sem_seg_{name}_step{step}"
    #         DatasetCatalog.register(
    #             all_name,
    #             lambda d=d: load_base_seg(d, root),
    #         )
    #         MetadataCatalog.get(all_name).set(
    #             image_root=image_dir,
    #             sem_seg_root=gt_dir,
    #             evaluator_type="IFS_sem_seg",
    #             ignore_label=255,
    #             **meta,
    #         )
    #     elif name=='train' and step == 1:
    #         all_name = f"voc_IFS_sem_seg_{name}_step{step}_shot_{shot}"
    #         DatasetCatalog.register(
    #             all_name,
    #             lambda d=d: load_novel_seg(d, task, shot, dataname='voc', root=root),
    #             # lambda split=split, name=name: load_fewshot_voc_seg(split, name),
    #         )

    #         MetadataCatalog.get(all_name).set(
    #             image_root=image_dir,
    #             sem_seg_root=gt_dir,
    #             evaluator_type="IFS_sem_seg",
    #             ignore_label=255,
    #             **meta,
    #         )
    #         # DatasetCatalog.register(
    #         #     all_name,
    #         #     lambda d=d: load_novel_seg(d, task, 1, 'voc', root),
    #         # )
    #         # MetadataCatalog.get(all_name).set(
    #         #     image_root=image_dir,
    #         #     sem_seg_root=gt_dir,
    #         #     evaluator_type="IFS_sem_seg",
    #         #     ignore_label=255,
    #         #     **meta,
    #         # )
    #         # all_name = f"voc_IFS_sem_seg_{name}_step{step}_shot_2"
    #         # DatasetCatalog.register(
    #         #     all_name,
    #         #     lambda d=d: load_novel_seg(d, task, 2, 'voc', root),
    #         # )
    #         # MetadataCatalog.get(all_name).set(
    #         #     image_root=image_dir,
    #         #     sem_seg_root=gt_dir,
    #         #     evaluator_type="IFS_sem_seg",
    #         #     ignore_label=255,
    #         #     **meta,
    #         # )
    #         # all_name = f"voc_IFS_sem_seg_{name}_step{step}_shot_5"
    #         # DatasetCatalog.register(
    #         #     all_name,
    #         #     lambda d=d: load_novel_seg(d, task, 5, 'voc', root),
    #         # )
    #         # MetadataCatalog.get(all_name).set(
    #         #     image_root=image_dir,
    #         #     sem_seg_root=gt_dir,
    #         #     evaluator_type="IFS_sem_seg",
    #         #     ignore_label=255,
    #         #     **meta,
    #         # )
    #     elif name=="val":
    #         all_name = f'voc_IFS_sem_seg_{name}'
    #         DatasetCatalog.register(
    #             all_name,
    #             lambda d=d: load_base_seg(d, root),
    #         )
    #         MetadataCatalog.get(all_name).set(
    #             image_root=image_dir,
    #             sem_seg_root=gt_dir,
    #             evaluator_type="IFS_sem_seg",
    #             ignore_label=255,
    #             **meta,
    #         )

    #     MetadataCatalog.get(all_name).set(
    #         image_root=image_dir,
    #         sem_seg_root=gt_dir,
    #         evaluator_type="IFS_sem_seg",
    #         ignore_label=255,
    #         **meta,
    #     )



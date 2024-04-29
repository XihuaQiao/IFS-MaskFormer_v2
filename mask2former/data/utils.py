import copy, random
import logging

from detectron2.data import detection_utils as utils

import numpy as np
import torch, tqdm, os, cv2
from torch.nn import functional as F


tasks = {
    "voc": {
        "offline":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                1: [],
            },
        "5-0":
            {
                0: [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                1: [1, 2, 3, 4, 5]
            },
        "5-1":
            {
                0: [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                1: [6, 7, 8, 9, 10]
            },
        "5-2":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
                1: [11, 12, 13, 14, 15]
            },
        "5-3":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                1: [16, 17, 18, 19, 20]
            },
    },
    "coco": {
        "offline":
            {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                    88, 89, 90],
                1: [],
            },
        "20-1":
            {
                0: [0, 1, 3, 4, 5, 7, 8, 9, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24,
                    25, 27, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 46, 48, 49, 50,
                    52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 65, 67, 72, 73, 74, 76, 77,
                    78, 80, 81, 82, 85, 86, 87, 89, 90],
                1: [2, 6, 10, 15, 19, 23, 28, 34, 38, 42, 47, 51, 55, 59, 63, 70, 75, 79, 84, 88]
            },
        "20-0":
            {
                0: [0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 24,
                    25, 28, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 47, 48, 49, 51,
                    52, 53, 55, 56, 57, 59, 60, 61, 63, 64, 65, 70, 72, 73, 75, 76, 77,
                    79, 80, 81, 84, 85, 86, 88, 89, 90],
                1: [1, 5, 9, 14, 18, 22, 27, 33, 37, 41, 46, 50, 54, 58, 62, 67, 74, 78, 82, 87]
            },
        "20-2":
            {
                0: [0, 1, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15, 17, 18, 19, 21, 22, 23,
                    25, 27, 28, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49, 50,
                    51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 67, 70, 73, 74, 75, 77,
                    78, 79, 81, 82, 84, 86, 87, 88, 90],
                1: [3, 7, 11, 16, 20, 24, 31, 35, 39, 43, 48, 52, 56, 60, 64, 72, 76, 80, 85, 89]
            },
        "20-3":
            {
                0: [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 22, 23,
                    24, 27, 28, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 48, 50,
                    51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 67, 70, 72, 74, 75, 76,
                    78, 79, 80, 82, 84, 85, 87, 88, 89],
                1: [4, 8, 13, 17, 21, 25, 32, 36, 40, 44, 49, 53, 57, 61, 65, 73, 77, 81, 86, 90],
            }
    }
}


def shot_generate(task, dataname):  # use_coco
    class_list = tasks[dataname]["offline"][0]
    base_list = tasks[dataname][task][0]
    novel_list = tasks[dataname][task][1]

    if dataname == "coco":
        mapper = {class_list[i]: i for i in range(81)}
        class_list = [i for i in range(81)]
        base_list = [mapper[i] for i in base_list]
        novel_list = [mapper[i] for i in novel_list]

    return class_list, base_list, novel_list


def load_base_seg(list_dir, root=None):
    assert os.path.isfile(list_dir), list_dir
    image_label_list, _ = torch.load(list_dir)
    dataset_dicts = []
    for img_path, gt_path in image_label_list:
        record = {}
        record["file_name"] = os.path.join(root, img_path)
        record["sem_seg_file_name"] = os.path.join(root, gt_path)
        dataset_dicts.append(record)
    return dataset_dicts


def load_novel_seg(list_dir, task, shot, ishot, dataname=None, root=None):
    print(shot)
    assert os.path.isfile(list_dir), list_dir
    _, novel_dict = torch.load(list_dir)

    _, _, novel_list = shot_generate(task, dataname)
    assert sorted(novel_dict.keys()) == sorted(novel_list), f"{sorted(novel_dict.keys())}"

    dataset_dicts = []

    # if shot == 0:
    #     for cls in novel_dict.keys():
    #         novel_images = novel_dict[cls]
    #         images_chosen = novel_images[ishot * 20: ishot * 20 + shot]
    #         for img_path, gt_path in images_chosen:
    #             record = {}
    #             record["novel_class"] = cls
    #             record["file_name"] = os.path.join(root, img_path)
    #             record["sem_seg_file_name"] = os.path.join(root, gt_path)
    #             dataset_dicts.append(record)
    # else:
    selected = []
    for cls in novel_dict.keys():
        novel_images = novel_dict[cls]
        l = []
        for (_, j) in novel_images:
            l.append(j.split('/')[1].split('.')[0])

        # print(f"ishot - {ishot}, nshot - {shot}, cls - {cls}, images - {len(novel_images)}")
        images_chosen = novel_images[ishot * 20: ishot * 20 + shot]
        selected.extend(l[ishot * 20: ishot * 20 + shot])
        for img_path, gt_path in images_chosen:
            record = {}
            record["novel_class"] = cls
            record["file_name"] = os.path.join(root, img_path)
            record["sem_seg_file_name"] = os.path.join(root, gt_path)
            dataset_dicts.append(record)
    
    random.shuffle(dataset_dicts)

    print(f"len of novel_seg dataset_dicts is {len(dataset_dicts)}")
    print(f"selected images - {selected}")

    return dataset_dicts


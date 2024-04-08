
import os

def add_seed(cfg):
    # if cfg.DATASETS.NAME == 'voc':
    #     if cfg.DATASETS.TASK == '5-0':
    #         seed = 4604572
    #     elif  cfg.DATASETS.SPLIT == 1:
    #         seed = 7743485
    #     elif  cfg.DATASETS.SPLIT == 2:
    #         seed = 5448843
    #     elif  cfg.DATASETS.SPLIT == 3:
    #         seed = 2534673
    # if cfg.DATASETS.dataname == 'coco':
    #     if cfg.DATASETS.SPLIT == 0:
    #         seed = 8420323
    #     elif  cfg.DATASETS.SPLIT == 1:
    #         seed = 27163933
    #     elif  cfg.DATASETS.SPLIT == 2:
    #         seed = 8162312
    #     elif  cfg.DATASETS.SPLIT == 3:
    #         seed = 3391510
    # if cfg.DATASETS.dataname == 'c2pv':
    #     seed = 321
    return ['SEED', 990920]

def add_dir(cfg):
    step = f'step{cfg.DATASETS.STEP}'

    if cfg.DATASETS.STEP == 1:
        OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'{cfg.DATASETS.NAME}_{cfg.DATASETS.TASK}', step, f'{cfg.DATASETS.SHOT}shot_i{cfg.DATASETS.iSHOT}')
    else:
        OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'{cfg.DATASETS.NAME}_{cfg.DATASETS.TASK}', step)

    return ['OUTPUT_DIR', OUTPUT_DIR]

def add_dataset(cfg):
    if cfg.DATASETS.STEP == 0:
        DATASETS_TRAIN = (f"{cfg.DATASETS.TRAIN[0]}_step{cfg.DATASETS.STEP}", )
    else:
        DATASETS_TRAIN = (f"{cfg.DATASETS.TRAIN[0]}_step{cfg.DATASETS.STEP}_shot_{cfg.DATASETS.SHOT}", )
    DATASETS_TEST = (cfg.DATASETS.TEST[0],)
    return ['DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST]
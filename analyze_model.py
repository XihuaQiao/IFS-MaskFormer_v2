import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:32"

import torch
torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=1, rank=0)


from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import build_model
from tqdm import tqdm
from mask2former.data.build import build_detection_test_loader, build_detection_train_loader
from mask2former.data.dataset_mappers.incremental_few_shot_mapper import IncrementalFewShotSemanticDatasetMapper
from train import build_train_loader, get_evaluator, setup

from detectron2.engine import default_argument_parser, launch
import numpy as np

from mask2former.data import register_all_coco, register_all_pascal

# from mask2former.data.datasets import register_voc_fewshotseg

def main(args):
    cfg = setup(args)

    num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

    if cfg.DATASETS.NAME == 'coco':
        register_all_coco(cfg.DATASETS.TASK, cfg.DATASETS.STEP, cfg.DATASETS.SHOT, cfg.DATASETS.iSHOT, cfg.DATASETS.EXTRA)
    elif cfg.DATASETS.NAME == 'voc':
        register_all_pascal(cfg.DATASETS.TASK, cfg.DATASETS.STEP, cfg.DATASETS.SHOT, cfg.DATASETS.iSHOT, cfg.DATASETS.EXTRA) 

    file_path = f"{cfg.OUTPUT_DIR}/indices-{cfg.DATASETS.NAME}-{cfg.DATASETS.TASK}.npy"

    if os.path.exists(file_path):
        print(f"using existing indices file from {file_path}")
        cal_metrix = np.load(file_path)
    else:
        model = build_model(cfg)
        dataset_name = cfg.DATASETS.TRAIN[0]
        mapper = IncrementalFewShotSemanticDatasetMapper(cfg, is_train=False)
        data_loader, _ = build_detection_test_loader(cfg, dataset_name, mapper)
        evaluator = get_evaluator(cfg, dataset_name, None)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
        )

        cal_metrix = np.zeros((num_classes, num_queries))

        for data in tqdm(data_loader):
            model(data, None)
            indices = model.criterion.indices[0]

            for (qIdxs, clses) in indices:
                for qIdx, cls in zip(qIdxs, clses):
                    cal_metrix[cls, qIdx] += 1
        
        np.save(file_path, cal_metrix)

    topK = args.topK
    strong_queries = set()
        
    for cls in range(num_classes):
        qIdxs = np.argsort(cal_metrix[cls, :])[:-(topK + 1):-1]
        print(f"class {cls}: ", end='')
        for qIdx in qIdxs:
            print(f"query {qIdx} -- {cal_metrix[cls, qIdx]}, ", end='')
            strong_queries.add(qIdx)
        print('')

    print(f"strong queries: {sorted(strong_queries)}")
    print(f"lenf of queries: {len(strong_queries)}")



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--topK', type=int, default=1, help="analyze top K uesd queries")

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
    


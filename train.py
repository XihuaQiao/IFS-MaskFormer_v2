import logging
import os, json

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:32"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from detectron2.modeling.postprocessing import sem_seg_postprocess
from mask2former.data.build import get_detection_dataset_dicts
from mask2former.data.utils import shot_generate


from collections import OrderedDict
import copy
import itertools
from typing import Any, Dict, List, Set
import torch
import random
import numpy as np

# we fix the random seed to 0,  this method can keep the results consistent in the same computer
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from tqdm import tqdm

import torch.nn.functional as F
import copy

import cv2

# torch.set_num_threads(4)
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23450', world_size=1, rank=0)
from torch.nn.parallel import DistributedDataParallel

from demo.predictor import DemoPredictor, VisualizationDemo 

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    # build_detection_test_loader,
    # build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

from mask2former.utils.addcfg import *
from mask2former import (
    IncrementalFewShotSemSegEvaluator,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from mask2former.data import (
    IncrementalFewShotSemanticDatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
    register_all_coco,
    register_all_pascal,
)

logger = logging.getLogger("detectron2")

def build_optimizer(cfg, model):
    weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
    weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR
    defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    if cfg.DATASETS.STEP == 0:
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
    else:
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if "backbone" in module_name:
                    value.requires_grad = False
                # if "class_embed.cls.1" in module_name:
                #     value.requires_grad = False
                # if "class_embed.cls.0" in module_name:
                #     value.requires_grad = False
                # if not "class_embed.cls.2" in module_name:
                #     value.requires_grad = False

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):     
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                # if "backbone" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 0.2
                # if "class_embed.cls.1" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 0.5
                # if "mask_embed" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 5
                # if "class_embed.cls.2" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 5
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})    

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # semantic segmentation
    if evaluator_type == 'IFS_sem_seg':
        evaluator_list.append(
            IncrementalFewShotSemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, test_loaders, evaluators):
    results = OrderedDict()
    for dataset_name, test_loader, evaluator in zip(cfg.DATASETS.TEST, test_loaders, evaluators):
        results_i = inference_on_dataset(model, test_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            assert isinstance(results_i, dict), "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def build_train_loader(cfg, predictor):
    mapper = IncrementalFewShotSemanticDatasetMapper(cfg, is_train=True, predictor=predictor)
    return build_detection_train_loader(cfg, mapper=mapper)

def do_train(cfg, model, model_old, predictor, resume=False, test_loaders = None, evaluators = None, cfg_old=None):

    model.train()
    if model_old:
        model_old.eval()
        DetectionCheckpointer(model_old, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    if cfg.DATASETS.STEP > 0:
        if comm.get_world_size() > 1:
            model.module.init_novel_stage()
        else:
            model.init_novel_stage()
    
    max_iter = cfg.SOLVER.MAX_ITER
    # max_iter = 100

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # if cfg.MODEL.MASK_FORMER.NUM_EXTRA_QUERIES > 0 and cfg.DATASETS.STEP > 0:
    #     print(f"Using Extra {cfg.MODEL.MASK_FORMER.NUM_EXTRA_QUERIES} Queries~~~~~~~~~~")
    #     model.module.init_novel_stage()

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    # data_loader = build_detection_train_loader(cfg)

    if cfg.DATASETS.STEP > 0 and not resume:
        # warm_up(cfg, model_old, model, distributed=comm.get_world_size() > 1, epoch=5)
        # cool_down(cfg_old, model_old, model, distributed=comm.get_world_size() > 1, epoch=1)

        # torch.save(model.state_dict(), f"pretrained/CosineCls-noClass-detach-{cfg.DATASETS.SHOT}shot-{cfg.DATASETS.iSHOT}.pth")
        d = f"pretrained/CosineCls-noClass-detach-{cfg.DATASETS.SHOT}shot-{cfg.DATASETS.iSHOT}.pth"
        print(f"loading pretrained model from {d}")
        if comm.get_world_size() > 1:
            weights = torch.load(d, map_location=model.module.device)
            model.module.load_state_dict(weights, strict=False)
        else:
            weights = torch.load(d, map_location=model.device)
            model.load_state_dict(weights, strict=False)
        if model_old:
            model_old.load_state_dict(weights, strict=True)
        if predictor:
            predictor.model.load_state_dict(weights, strict=True)

        # feat = model.sem_seg_head.predictor.class_embed.cls[0].weight.data
        # model_old.sem_seg_head.predictor.imprint_weights_step(feat, -1)
        # feat = model.sem_seg_head.predictor.class_embed.cls[1].weight.data
        # model_old.sem_seg_head.predictor.imprint_weights_step(feat, 0)

        # torch.save(model_old.state_dict(), f"pretrained/CosineCls.pth")
        # print(f"finish saving~")

        # model.train()

    train_loader = build_train_loader(cfg, predictor)
    logger.info("Starting training from iteration {}".format(start_iter))

    best_eval = 0

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(train_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            if model_old:
                with torch.no_grad():
                    outputs_old = model_old(data)
            else:
                outputs_old = None
            optimizer.zero_grad()

            loss_dict = model(data, outputs_old)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                # and iteration > start_val
            ):
                results = do_test(cfg, model, test_loaders, evaluators)
                comm.synchronize()

                if results:   
                    results_dict = dict(results)['sem_seg']
                    storage.put_scalars(**results_dict, smoothing_hint=False)
                    if cfg.DATASETS.STEP == 0:
                        cur_eval = results['sem_seg']['mIoU']
                    else:
                        cur_eval = results['sem_seg']['hmIoU']

                    if cur_eval > best_eval:
                        best_eval = cur_eval
                        periodic_checkpointer.save('model_best')

                        results_dict = json.dumps(results_dict)
                        f = open(os.path.join(cfg.OUTPUT_DIR, 'best.json'), 'w')
                        f.write(results_dict)
                        f.close()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

    # import pandas as pd
    # pd.DataFrame(model.module.query_updates).to_csv(f"{cfg.OUTPUT_DIR}/query_updates.csv")
    # pd.DataFrame(model.module.base_query_updates).to_csv(f"{cfg.OUTPUT_DIR}/base_query_updates.csv")
    # pd.DataFrame(model.module.novel_query_updates).to_csv(f"{cfg.OUTPUT_DIR}/novel_query_updates.csv")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    # 
    cfg.merge_from_list(add_seed(cfg))
    cfg.merge_from_list(add_dir(cfg)) 
    cfg.merge_from_list(add_dataset(cfg))

    if not cfg.MODEL.WEIGHTS: cfg.MODEL.WEIGHTS = f"{cfg.OUTPUT_DIR[:-1]}0/model_best.pth"
    if not cfg.MODEL.CONFIG: cfg.MODEL.CONFIG = cfg.MODEL.WEIGHTS.replace("model_best.pth", "config.yaml")

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def setupOld(file, model_weights):
    assert file, "ERROR: when using Knowledge Distillation, cfg.MODEL.CONFIG shoule be provided!"
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return cfg


def cool_down(cfg_old, model_old, model, distributed, epoch=5):
    # model_old = model_old.module if distributed else model_old
    model = model.module if distributed else model
    device = model.device
    
    _, base_classes, novel_classes = shot_generate(cfg_old.DATASETS.TASK, cfg_old.DATASETS.NAME)
    # base_classes = base_classes[1:]     # 排除bkg类别
    print(f"base_classes - {base_classes}")

    sum_features = torch.zeros(len(base_classes), cfg_old.MODEL.MASK_FORMER.HIDDEN_DIM).to(device)

    mapper = IncrementalFewShotSemanticDatasetMapper(cfg_old, is_train=False)
    _, dataset = build_detection_test_loader(cfg_old, cfg_old.DATASETS.TRAIN[0], mapper=mapper)

    count = torch.zeros(len(base_classes)).to(device)
    for ep in range(epoch):
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"epoch {ep}: "):
                data = dataset[idx]
                out_size = data['image'].shape[-2:]
                out = F.interpolate(model_old([data])['multi_scale_features'][0], size=out_size, mode='bilinear', align_corners=False).squeeze_(0)
                
                for i, cl in enumerate(data['instances'].gt_classes):
                    # if cl == 0: continue
                    mask = copy.deepcopy(data['instances'].gt_masks[i]).to(device).float()
                    feat = out[:, mask == 1.0]
                    feat = F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                    for j, _ in enumerate(base_classes):
                        if _ == cl: _idx = j
                    sum_features[_idx] += feat
                    count[_idx] += 1

                    # del mask
    
    assert torch.any(count != 0), "Error, a base class has no pixels"
    features = F.normalize(sum_features, dim=1)
    print(f"cool down! at step {cfg_old.DATASETS.STEP}")
    model.sem_seg_head.predictor.imprint_weights_step(step=-1, features=features[0:1, ...])
    model.sem_seg_head.predictor.imprint_weights_step(step=cfg_old.DATASETS.STEP, features=features[1:, ...])


def warm_up(cfg, model_old, model, distributed, epoch=5):
    # model_old = model_old.module if distributed else model_old
    model = model.module if distributed else model
    # model_old.eval()
    device = model.device
    _, base_classes, novel_classes = shot_generate(cfg.DATASETS.TASK, cfg.DATASETS.NAME)
    print(f"novel_classes - {novel_classes}")
    sum_features = torch.zeros(len(novel_classes), cfg.MODEL.MASK_FORMER.HIDDEN_DIM).to(device)

    mapper = IncrementalFewShotSemanticDatasetMapper(cfg, is_train=False)
    _, dataset = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0], mapper = mapper)

    count = torch.zeros(len(novel_classes)).to(device)
    for ep in range(epoch):
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"epoch {ep}: "):
                data = dataset[idx]
                '''
                out = model([data])['multi_scale_features'][0]
                out_size = data['image'].shape[-2:]     # (512, 512)
                out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=False).squeeze_(0)
                # out += F.interpolate(_features[-3], size=out_size, mode='bilinear', align_corners=False).squeeze_(0)
                # out = sem_seg_postprocess(out, out_size, data['height'], data['width'])

                '''

                out_size = data['image'].shape[-2:]
                out = F.interpolate(model_old([data])['multi_scale_features'][0], size=out_size, mode='bilinear', align_corners=False).squeeze_(0)

                cl = data['novel_class']
                for i, _ in enumerate(data['instances'].gt_classes):
                    if _ == cl: _idx = i
                mask = copy.deepcopy(data['instances'].gt_masks[_idx]).to(device).float()
                
                # from mask2former.data.datasets.register_voc_fewshotseg import CLASS_NAMES
                # _img = np.int8(mask.cpu().numpy())
                # _img = _img * 255
                # cv2.imwrite(f"./imgs/{data['file_name'].split('/')[-1].split('.')[0]}_{CLASS_NAMES[cl]}_mask.png", _img)
                # _img = data['image'].cpu().numpy().transpose(1, 2, 0)
                # _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f"./imgs/{data['file_name'].split('/')[-1].split('.')[0]}_{CLASS_NAMES[cl]}.jpg", _img)

                feat = out[:, mask == 1.0]  # D * F
                feat = F.normalize(F.normalize(feat, dim=0).sum(dim=1), dim=0)
                for i, _ in enumerate(novel_classes):
                    if _ == cl: _idx = i
                sum_features[_idx] += feat
                count[_idx] += 1

    assert torch.any(count != 0), "Error, a novel class has no pixels"
    features = F.normalize(sum_features, dim=1)
    print(f"warm up! at step {cfg.DATASETS.STEP}")
    model.sem_seg_head.predictor.imprint_weights_step(step=cfg.DATASETS.STEP, features=features)


def main(args):
    cfg = setup(args)

    if cfg.DATASETS.NAME == 'coco':
        register_all_coco(cfg.DATASETS.TASK, cfg.DATASETS.STEP, cfg.DATASETS.SHOT, cfg.DATASETS.iSHOT, cfg.DATASETS.EXTRA)
    elif cfg.DATASETS.NAME == 'voc':
        register_all_pascal(cfg.DATASETS.TASK, cfg.DATASETS.STEP, cfg.DATASETS.SHOT, cfg.DATASETS.iSHOT, cfg.DATASETS.EXTRA)

    model = build_model(cfg)

    # build test set first
    test_loaders, evaluators = [], []
    for dataset_name in cfg.DATASETS.TEST:
        mapper = IncrementalFewShotSemanticDatasetMapper(cfg, is_train=False)
        test_loader, _ = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        evaluator = get_evaluator(cfg, dataset_name, None)
        test_loaders.append(test_loader)
        evaluators.append(evaluator)

    # logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, test_loaders, evaluators)

    cfg_old = setupOld(cfg.MODEL.CONFIG, cfg.MODEL.WEIGHTS) if cfg.MODEL.KD.ENABLE or cfg.MODEL.MASK_FORMER.PSEUDO_LABEL else None
    model_old = build_model(cfg_old) if cfg.MODEL.KD.ENABLE else None
    predictor = DemoPredictor(cfg_old) if cfg.MODEL.MASK_FORMER.PSEUDO_LABEL else None

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, model_old, predictor, resume=args.resume, test_loaders=test_loaders, evaluators=evaluators, cfg_old=cfg_old)

    # return do_test(cfg, model, test_loaders, evaluators)


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.num_gpus == 1:
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=1, rank=0)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

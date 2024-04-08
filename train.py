import logging
import os, json

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:32"

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

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):     
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                # if "proposal_head.0" in module_name or "proposal_head.1" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 0.1
                # if "backbone" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 0.1
                # if "predictor.transformer" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * 0.1
                # if "pixel_decoder" in module_name:
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


def do_train(cfg, model, model_old, predictor, resume=False, test_loaders = None, evaluators = None):
    
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

    # for module_name, module in model.named_modules():
    #     for module_param_name, value in module.named_parameters(recurse=False):
    #         if not value.requires_grad:
    #             print(f"{module_name} - {module_param_name}")

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    if cfg.DATASETS.STEP > 0:
        if comm.get_world_size() > 1:
            model.module.init_novel_stage()
        else:
            model.init_novel_stage()
    
    # max_iter = cfg.SOLVER.MAX_ITER
    max_iter = 200

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
                (iteration + 1) % 10 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


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


def setupOld(file):
    assert file, "ERROR: when using Knowledge Distillation, cfg.MODEL.CONFIG shoule be provided!"
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(file)
    cfg.freeze()
    return cfg


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
        test_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        evaluator = get_evaluator(cfg, dataset_name, None)
        test_loaders.append(test_loader)
        evaluators.append(evaluator)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, test_loaders, evaluators)

    cfg_old = setupOld(cfg.MODEL.CONFIG) if cfg.MODEL.KD.ENABLE or cfg.MODEL.MASK_FORMER.PSEUDO_LABEL else None
    model_old = build_model(cfg_old) if cfg.MODEL.KD.ENABLE else None
    predictor = DemoPredictor(cfg_old) if cfg.MODEL.MASK_FORMER.PSEUDO_LABEL else None

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, model_old, predictor, resume=args.resume, test_loaders=test_loaders, evaluators=evaluators)
    # return do_test(cfg, model)


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

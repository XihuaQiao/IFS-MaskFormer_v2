#!/usr/bin/env bash

# PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
GPUS=4

cd /media/data/qxh/workspace/EDTER
python -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=29501 \
    tools/test_local.py --launcher pytorch ${@:1}\
    --globalconfig configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py --config configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py --checkpoint pretrain/iter_40000.pth --global-checkpoint pretrain/iter_30000.pth --tmpdir COCO_TRAIN_result


# 修改了 EDTER/mmseg/models/segmentors/encoder_decoder_local8x8.py 中slide_inference函数，让图片先pad 0，再分割，满足EDTER预训练模型的320x320和160x160的设定
# 修改 _base_ 中文件
#   /media/data/qxh/workspace/EDTER/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py
#   /media/data/qxh/workspace/EDTER/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py
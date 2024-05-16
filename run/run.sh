#!/bin/bash

# bash run/train.sh 0,1,2,3 4 out/voc-R101-ClsAftSeg-noClass/voc_5-3/step0/model_best.pth out/voc-R101-ClsAftSeg-noClass/nothing

bash run/train.sh 0,1,2,3 4 out/voc-R101-CosineCls-noClass-detach-1/voc_5-3/step0/model_best.pth out/voc-R101-CosineCls-noClass-detach-1/extra_query

# bash run/train.sh 3 1 out/voc-R101-CosineCls-clsAftSeg/voc_5-3/step0/model_best.pth out/voc-R101-CosineCls-clsAftSeg/finetune

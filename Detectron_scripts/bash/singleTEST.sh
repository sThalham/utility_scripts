#!/bin/bash

xterm -e "nvidia-smi -l 1" &
xterm -e "htop" &

python2 /home/sthalham/git/Detectron-Tensorflow/tools/train_net.py --cfg /home/sthalham/workspace/proto/Detectron_scripts/Detectron_configs/tlessArti09052018.yaml OUTPUT_DIR /home/sthalham/data/T-less_Detectron/output/tlessArti09052018_V2


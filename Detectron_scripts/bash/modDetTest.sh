#!/bin/bash

xterm -e "nvidia-smi -l 1" &
xterm -e "htop" &

python2 /home/sthalham/git/Detectron-tf_mod/tools/train_net.py --cfg /home/sthalham/workspace/proto/Detectron_scripts/Detectron_configs/linemodArtiHAA.yaml OUTPUT_DIR /home/sthalham/data/T-less_Detectron/output/linemodArti09062018


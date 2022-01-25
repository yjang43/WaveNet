#!/bin/bash

# load data
cp /staging/yjang43/dataset.npz ./

# # install requirements
# pip install requirements.txt
python main.py --epoch 10 --lr 0.001 --batch_sz 128 --max_itr 10000 \

# remove data
rm  dataset.npz

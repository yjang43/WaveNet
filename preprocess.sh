#!/bin/bash

# load data
cp /staging/yjang43/vctk.zip ./
unzip -q vctk.zip -d vctk

# install requirements
pip install requirements.txt
python main.py --data_dir vctk/wav48_silence_trimmed/p228/ \
               --dataset_path dataset \

# remove data
mv ./dataset.npz /staging/yjang43/dataset.npz
rm -r vctk.zip vctk


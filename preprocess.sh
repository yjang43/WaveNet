#!/bin/bash

# load data
cp /staging/yjang43/[DATA_NAME].tar.gz ./
tar -xzvf [DATA_NAME].tar.gz

# # install requirements
# pip install requirements.txt
python main.py --data_dir  \
               --dataset_path \

# remove data
rm -r [DATA_NAME].tar.gz [DATA_NAME]

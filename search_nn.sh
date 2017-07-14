#!/bin/bash

for i in `seq 0 3 30`; do
    CUDA_VISIBLE_DEVICES=3 python find_NN.py --experiment debug/ --modelName ${i}
done


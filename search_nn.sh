#!/bin/bash

for i in `seq 13 1 99`; do
    CUDA_VISIBLE_DEVICES=5 python find_NN.py --experiment noise_shuffle_bug/ --modelName ${i}
done


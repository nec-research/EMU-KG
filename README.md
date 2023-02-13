# MUTUP – A new Efficient Negative Sampling Method For Knowledge-Graph Link Prediction

## Introduction

This repository provides the PyTorch implementation of _MUTUP_ technique presented in _MUTUP – A new Efficient Negative Sampling Method For Knowledge-Graph Link Prediction_ paper as well as several popular KGE models.

## Execution

As an example, the following command trains and validates a TransE model on wn18rr dataset by using Mutup with uniform negative samplingd:

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --cuda \
    --do_train --do_valid \
    --data_path data/wn18rr \
    --model TransE \
    -n 256 -b 1024 -d 1000 \
    -g 24.0 -a 1.0 \
    -lr 0.0001 --max_steps 150000 \
    -save models/TransE_wn18rr_0 --test_batch_size 16 \
    -khop 3 -nrw 1000 \ 
    -if_CE 1 -if_Mutup 0.3 -neg_label 0.5 -CE_coef 0.1
```

To check all the available arguments, you can run `python codes/run.py --help`.

## Reproducibility

To reproduce the results presented in the ARR 2023 Feb. paper _MUTUP – A new Efficient Negative Sampling Method For Knowledge-Graph Link Prediction_, you can use the commands provided in `experiments_Mutup.sh`.

## Infrustrucutre

All experiments were carried on a server with one NVIDIA GeForce GTX 1080 Ti GPU.

## Acknowledgments

Our implemention is based on the PyTorch implementation of _Structure Aware Negative Sampling in Knowledge Graphs_ provided [here](https://github.com/kahrabian/SANS).

# EMU – Efficient Negative Sampling Method For Knowledge-Graph Link Prediction

![EMU schematic picture](/EMU-schematic.png)

## Introduction

This repository provides the PyTorch implementation of _EMU_ technique presented in _Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction_ paper as well as several popular KGE models.

## Execution

As an example, the following command trains and validates a TransE model on wn18rr dataset by using EMU with uniform negative sampling:

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

To reproduce the results presented in the TMLR paper _Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction_, you can use the commands provided in `experiments_EMU.sh`.

The [Mamba](https://mamba.readthedocs.io/en/latest/index.html) environment can be reproduced using `environment.yml` as: 

```
mamba env create -f environment.yml
```

## Infrastructure

All experiments were carried on a server with one NVIDIA GeForce GTX 1080 Ti GPU.

## Acknowledgments

Our implementation is based on the PyTorch implementation of _Structure Aware Negative Sampling in Knowledge Graphs_ provided [here](https://github.com/kahrabian/SANS).

## Citation

```bibtex
@article{
takamoto2025optimal,
title={Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction},
author={Makoto Takamoto and Daniel Onoro Rubio and Wiem Ben Rim and Takashi Maruyama and Bhushan Kotnis},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=B4SyciDyIh},
note={}
}
```
```bibtex
@misc{takamoto2025optimalembeddingguidednegative,
      title={Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction}, 
      author={Makoto Takamoto and Daniel Oñoro-Rubio and Wiem Ben Rim and Takashi Maruyama and Bhushan Kotnis},
      year={2025},
      eprint={2504.03327},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.03327}, 
}
```
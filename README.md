Meta Propagation Networks for Graph Few-shot Semi-supervised Learning (AAAI2022)
============

## Meta-LP

This is the source code and data of AAAI2022 paper "Meta Propagation Networks for Graph Few-shot Semi-supervised Learning"


## Requirements
python==3.6.6 

torch==1.4.0

## Datasets

We use four datasets: Cora_ml, Citeseer, Pubmed and ms_academic adopted from [APPNP](https://github.com/klicperajo/ppnp).

## Usage
To run Meta-PN on 5-shot:
```
python train.py --dataset cora_ml --shot 5
python train.py --dataset citeseer --shot 5 --lr_main 0.05 
python train.py --dataset pubmed --shot 5 --batch_size 4096
python train.py --dataset ms_academic --shot 5 --batch_size 4096 --init_alpha 0.2 --lr_main 0.01
```
## Citation

Please cite our paper if you use this code in your own work:

```
@InProceedings{ding2022meta,
  title     = {Meta Propagation Networks for Graph Few-shot Semi-supervised Learning},
  author    = {Ding, Kaize and Wang, Jianling and Caverlee, James and Liu, Huan},
  booktitle = {AAAI},
  year     = {2022},
}
```

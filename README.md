# Implementation of Prompt Based Incremental Learning Using Attention Diversity

This repository contains PyTorch implementation code for the KSC 2022 paper: **Prompt Based Incremental Learning Using Attention Diversity**

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, clone the repository locally:
```
git clone https://github.com/Lee-JH-KR/CapstoneDesign2
cd CapstoneDesign2
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100, pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `datasets.py` as follows

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

## Training
To train a model via command line:

Single node with single gpu
```
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_l2p_diverse \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5
```

Single node with multi gpus
```
python -m torch.distributed.launch \
        --nproc_per_node=<Num GPUs> \
        --use_env main.py \
        cifar100_l2p_diverse \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5
```

Also available in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system by changing options on `train_cifar100_l2p.sh` or `train_five_datasets.sh` properly.

### Multinode train

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train a model on 2 nodes with 4 gpus each:

```
python run_with_submitit.py cifar100_l2p_diverse --shared_folder <Absolute Path of shared folder for all nodes>
```

Absolute Path of shared folder must be accessible from all nodes.<br>
According to your environment, you can use `NCLL_SOCKET_IFNAME=<Your own IP interface to use for communication>` optionally.

## Evaluation
To evaluate a trained model:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p_diverse --eval
```

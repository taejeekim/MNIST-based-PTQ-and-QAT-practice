# MNIST-based-PTQ-and-QAT-practice
This repository is for practice Quantization specifically PTQ (Post-training Quantization) and QAT (Quantization-aware training) with MNIST dataset

## Setting
```bash
conda create -n quant python=3.10; conda activate quant
pip install jupyterlab matplotlib tqdm
# install pytorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # You should install apropriate pytorch from https://pytorch.org/get-started/locally/
```

This repository is for practice Quantization specifically PTQ (Post-training Quantization) and QAT (Quantization-aware training) with MNIST dataset

## Setting
```bash
conda create -n quant python=3.10; conda activate quant
pip install jupyterlab matplotlib tqdm
#install Pytorch (see below)
```

### How to install appropriate pytorch?
1. check your CUDA version
```bash
# You can check your CUDA version by below command
nvcc --version
```
2. Go to pytroch installation site and click appropriate setting
<img width="2002" height="1606" alt="image" src="https://github.com/user-attachments/assets/9263a3b2-b0b1-4ea4-bf6c-293a200e44d8" />

3. install pytorch
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader =DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
calib_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# From scratch ResNet18 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1 # ResNet18은 Botteleneck 구조가 아님

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu= nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # Skip connection을 위한 downsample
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
    

# From scratch ResNet18 (MNIST 맞춤)
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels =64

        # 초기 conv (MNIST: 1채널 입력)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # maxpool 생략 (MNIST 이미지 작아서 feature map 유지)

        # Residual layers (ResNet18: [2, 2, 2, 2] blocks)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Avarage Pool + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride !=1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        
        layers= []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    

# 평가/ 크기/ 시간 함수
def evaluate(model, loader=test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total * 100

def get_float_model_size(model):
    torch.save(model.state_dict(), "tmp.pth")
    size_mb= os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    return size_mb

def get_quantized_model_size(model, weight_scales):
    q_state_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            scale = weight_scales[name]
            q_weight = torch.round(param.data / scale).clamp(-128, 127).to(torch.int8) # Real INT
            q_state_dict[name + '.quantized_weight'] = q_weight
            q_state_dict[name + '.scale'] = torch.tensor(scale)
        else:
            q_state_dict[name] = param.data # bias etc. is float
    torch.save(q_state_dict, "tmp_quant.pth")
    size_mb = os.path.getsize("tmp_quant.pth") / 1e6
    os.remove("tmp_quant.pth")
    return size_mb

def measure_inference_time(model, num_samples=1000):
    model.eval()
    dummy = torch.randn(num_samples, 1,28, 28).to(device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    return (time.time() - start) / num_samples * 1000
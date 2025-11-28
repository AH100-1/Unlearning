import torch
import torch.optim as optim  
import torch.nn as nn 
import torchvision
from torchvision.models import resnet34, resnet18


class custom_res18(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # ResNet18 모델 로드 (pretrained=False로 처음부터 학습)
        self.model = resnet18(pretrained=False)
        
        # CIFAR-10을 위한 첫 번째 블록 수정
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # 분류기를 CIFAR-10 클래스 수에 맞게 변경
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
           
    def forward(self, x):
        return self.model(x)
    


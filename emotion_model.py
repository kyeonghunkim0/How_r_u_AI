import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# 1) EmotionCNN 정의
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32,64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,32,24,24]
        x = self.pool(F.relu(self.conv2(x)))  # [B,64,12,12]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)             # flatten → [B, 64*12*12]
        return x

# 2) HybridEmotionModel 정의 (EmotionCNN + ResNet18 결합)
class HybridEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (1) custom CNN 백본
        self.cnn = EmotionCNN()

        # (2) ResNet18 백본 (흑백 입력, 분류 레이어 제거)
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)
        self.resnet.fc = nn.Identity()  # 최종 FC를 제거하고 feature만 반환

        # (3) 두 feature를 합친 뒤 감정 분류기
        self.fc = nn.Sequential(
            nn.Linear((64*12*12) + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        # x1: EmotionCNN → [B, 9216],  x2: ResNet18 → [B, 512]
        x1 = self.cnn(x)
        x2 = self.resnet(x)
        x  = torch.cat((x1, x2), dim=1)  # [B, 9728]
        return self.fc(x)

# —— 위 정의 셀을 실행하신 뒤에는 아래 코드만 실행하세요 ——
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)      # 입력 채널 1 (Grayscale), 출력 채널 32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)          # 48x48 이미지 → 2번 pooling → 12x12
        self.fc2 = nn.Linear(128, 7)                     # 7가지 감정 클래스

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))             # [B, 32, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))             # [B, 64, 12, 12]
        x = self.dropout(x)
        x = x.view(-1, 64 * 12 * 12)                     # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# model = EmotionCNN()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# from torchvision.datasets import ImageFolder
# from torchvision import transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_dataset = ImageFolder(root='train', transform=transform)
# val_dataset = ImageFolder(root='test', transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# num_epochs = 5

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# from torchvision.models import resnet18

# resnet = resnet18(pretrained=False)  # 사전 학습 weight 없이 시작
# resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# resnet.fc = nn.Linear(resnet.fc.in_features, 7)

# model = resnet.to(device)

# import torch.optim as optim
# import torch.nn as nn

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")

# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Validation Accuracy: {100 * correct / total:.2f}%")

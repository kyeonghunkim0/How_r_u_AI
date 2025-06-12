from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# ——— HybridEmotionModel 정의 ———
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

class HybridEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = EmotionCNN()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.resnet.fc   = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear((64*12*12) + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.resnet(x)
        x  = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# ——— 모델 불러오기 ———
model = HybridEmotionModel()
model.load_state_dict(torch.load("model4.pth", map_location='mps'))
model.eval()

# ——— 감정 라벨 정의 ———
labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ——— Flask 앱 초기화 ———
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'No image uploaded'}), 400

    try:
        # 이미지 로드 및 흑백 변환
        image = Image.open(request.files['image']).convert('L')
    except Exception:
        return jsonify({'error':'Invalid image file'}), 400

    # 전처리: 크기 조정 → Tensor → 정규화
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    # 추론
    with torch.no_grad():
        logits = model(input_tensor)                           # raw outputs
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()# [p1, p2, ..., p7]
        top_idx = int(torch.argmax(logits, dim=1).item())

    # 클래스별 확률(%) 분포
    distribution = {
        lbl: round(p * 100, 1)
        for lbl, p in zip(labels, probs)
    }

    result = {
        'top_emotion':    labels[top_idx],
        'top_confidence': distribution[labels[top_idx]],
        'distribution':   distribution
    }
    return jsonify(result)

if __name__ == '__main__':
    # 0.0.0.0 바인딩 + 포트 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
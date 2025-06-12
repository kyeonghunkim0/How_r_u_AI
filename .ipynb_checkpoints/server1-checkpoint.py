from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

# ✅ ResNet18 모델 정의 및 수정
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("model1.pth", map_location='cpu'))  # 저장된 ResNet18 기반 모델 weight
model.eval()

# ✅ 감정 라벨 (FER2013 기준)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Flask 앱 초기화
app = Flask(__name__)

# ✅ /predict API 라우트
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = Image.open(request.files['image']).convert('L')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    # ✅ 전처리: 흑백 + 정규화 + 크기 조정
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0][predicted_idx].item()

    result = {
        'emotion': labels[predicted_idx],
        'confidence': round(confidence, 3) * 100
    }

    return jsonify(result)

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(debug=True)

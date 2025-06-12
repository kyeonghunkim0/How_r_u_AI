from flask import Flask, request, jsonify  # Flask 웹 서버 구축용
from PIL import Image                    # 이미지 처리용
import torch                             # PyTorch
import torchvision.transforms as transforms  # 이미지 전처리
import io                                # 바이트 스트림 처리용

# 사용자 정의 모델 클래스 import
from make_cnn_model import EmotionCNN

# ✅ 모델 불러오기
model = EmotionCNN()
model.load_state_dict(torch.load("model.pth", map_location='cpu'))  
model.eval()  # 추론 모드로 설정

# ✅ 감정 라벨 정의 (FER2013 기준)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Flask 앱 초기화
app = Flask(__name__)

# ✅ /predict API 라우트 정의
@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 파일이 없을 경우
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # 이미지 파일을 열어 흑백으로 변환
        image = Image.open(request.files['image']).convert('L')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    # ✅ 전처리: 모델 입력 형식에 맞게 변환
    transform = transforms.Compose([
        transforms.Resize((48, 48)),              # 크기 통일
        transforms.ToTensor(),                    # Tensor 변환
        transforms.Normalize((0.5,), (0.5,))      # 정규화
    ])
    input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    # ✅ 추론
    with torch.no_grad():
        output = model(input_tensor)  # 모델에 이미지 입력
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 확률 계산
        predicted_idx = torch.argmax(probabilities, 1).item()       # 가장 확률 높은 클래스
        confidence = probabilities[0][predicted_idx].item()         # 신뢰도 추출

    # ✅ 결과 구성
    result = {
        'emotion': labels[predicted_idx],
        'confidence': round(confidence, 3) * 100  # 소수점 3자리 반올림에 100 곱해서 퍼센트로 출력
    }

    return jsonify(result)  # JSON 응답

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(debug=True)

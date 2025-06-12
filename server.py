# server.py
import io
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from emotion_model import HybridEmotionModel  # 감정 분류 모델 정의 모듈

app = Flask(__name__)

# ─── 얼굴 검출 모델 로드 ─────────────────────
# ultralytics hub에서 자동으로 다운로드되는 스템 이름(yolov8n.pt) 사용
yolo_model = YOLO("yolov8n.pt")
yolo_model.fuse()  # Optional: 속도 최적화를 위해 레이어 융합

# ─── 감정 분류 모델 로드 ─────────────────────
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
emotion_model = HybridEmotionModel().to(device)
# 1) FER2013 사전학습 가중치 로드
emotion_model.load_state_dict(torch.load("fer_pretrained.pth", map_location=device))
# 2) RAF-DB 파인튜닝 가중치 로드
emotion_model.load_state_dict(torch.load("raf_finetuned.pth",   map_location=device))
emotion_model.eval()

# ─── 이미지 전처리 정의 ─────────────────────
transform = T.Compose([
    T.Resize((48, 48)),
    T.Grayscale(),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])
EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",     5: "Surprise",
    6: "Neutral"
}

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # 1) 바이너리 → OpenCV 이미지
    data = file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 2) 얼굴 검출
    results = yolo_model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    face_count = len(boxes)

    # 3) 얼굴 박스 그리기
    annotated = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4) 얼굴이 1개일 때만 감정 분석
# 4) 얼굴이 하나라도 있으면 첫 번째 얼굴로 감정 분석 + 확률 분포
    emotion_info = None
    if face_count >= 1:
    # 첫 번째 얼굴 영역 크롭
        x1, y1, x2, y2 = boxes[0]
        face_crop = annotated[y1:y2, x1:x2]
        face_pil  = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        inp = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = emotion_model(inp)                            
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]   
    
        # 퍼센트로 변환 & 반올림(소수점 한 자리)
        percents = [round(float(p) * 100, 1) for p in probs]
        top_idx   = int(np.argmax(percents))
        top_label = EMOTION_LABELS[top_idx]
        top_conf  = percents[top_idx]  # 예: 87.3
    
    # 클래스별 퍼센트 dict 생성 (float)
    prob_dict = {
        EMOTION_LABELS[i]: percents[i] for i in range(len(percents))
    }
    
    emotion_info = {
        "label":       top_label,    # 예: "Happy"
        "confidence":  top_conf,     # 예: 87.3
        "probabilities": prob_dict   # 예: {"Happy": 87.3, ...}
    }



    success, buffer = cv2.imencode('.jpg', annotated)
    if not success:
        return jsonify({"error": "Image encoding failed"}), 500
    b64_str = base64.b64encode(buffer.tobytes()).decode('utf-8')

    # 6) JSON 응답
    return jsonify({
        "face_count": face_count,
        "emotion": emotion_info,
        "annotated_image": b64_str
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

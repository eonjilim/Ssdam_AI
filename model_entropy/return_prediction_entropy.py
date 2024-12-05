import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import sys

# 모델 클래스 정의 및 불러오기
def load_model(model_path="best_trained_model_v2.pth"):
    model = timm.create_model('efficientnet_b0', pretrained=False)  # 미리 학습된 가중치는 불러오지 않음
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512), 
        nn.ReLU(), 
        nn.Dropout(0.5), 
        nn.Linear(512, 19)  # 클래스 개수 (학습 시 설정한 클래스 개수와 동일)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 설정
    return model

# 이미지 데이터 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 표준화
    ])
    image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 변환
    return transform(image).unsqueeze(0)  # 배치 차원을 추가

# 예측 함수
def predict(image_path, model_path="best_trained_model_v2.pth"):
    # 모델 로드
    model = load_model(model_path)
    # 이미지 전처리
    input_tensor = preprocess_image(image_path)
    # 예측
    with torch.no_grad():
        outputs = model(input_tensor)  # 모델에 입력
        probabilities = torch.softmax(outputs, dim=1)  # 소프트맥스 확률 계산
        predicted_class = torch.argmax(probabilities, dim=1).item()  # 가장 높은 확률 클래스
        confidence = probabilities[0, predicted_class].item()  # 해당 클래스의 확률
        
    return predicted_class, confidence

# 품목명 리스트 (클래스 번호와 품목명을 매핑)
class_names = [
    "품목1", "품목2", "품목3", "품목4", "품목5", "품목6", "품목7", "품목8", "품목9", "품목10",
    "품목11", "품목12", "품목13", "품목14", "품목15", "품목16", "품목17", "품목18", "품목19"
    # 실제 학습 데이터의 품목명 리스트로 바꾸세요
]

# 메인 함수
if __name__ == "__main__":
    try:
        # 사용자로부터 입력 받기
        image_path = sys.argv[1]  # 첫 번째 인자: 이미지 경로
        model_path = sys.argv[2]  # 두 번째 인자: 모델 경로
        
        # 예측 수행
        pred_class, conf = predict(image_path, model_path)
        
        # 예측된 품목명
        predicted_class_name = class_names[pred_class]
        
        # 결과 출력
        print(f"Predicted Class: {predicted_class_name}, Confidence: {conf:.2f}")  # 품목명 및 신뢰도 출력
    except Exception as e:
        print(f"Error: {str(e)}")
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F  # 엔트로피 계산을 위한 함수 임포트

# 모델 클래스 정의 및 불러오기
def load_model(model_path="best_trained_model_v6.pth"):
    model = timm.create_model('efficientnet_b0', pretrained=False)  # 미리 학습된 가중치는 불러오지 않음
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512), 
        nn.ReLU(), 
        nn.Dropout(0.5), 
        nn.Linear(512, 18)  # 클래스 개수 (학습 시 설정한 클래스 개수와 동일)
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

# 엔트로피 계산 함수
def calculate_entropy(probabilities):
    # 엔트로피 = - Σ p(x) * log(p(x))
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy.item()  # 단일 값 반환

# 예측 함수 (엔트로피와 클래스 출력)
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
        
        # 엔트로피 계산
        entropy = calculate_entropy(probabilities)

    return predicted_class, entropy

# 품목명 리스트 (클래스 번호와 품목명을 매핑)
class_names = [
    "TV 받침", "거울", "빨래건조대", "서랍장", "쇼파", "시계", "안마의자", "의자", "자전거", "장롱", "장식장", "책꽂이", "책상", "책장", "침대", "테이블", "피아노", "화장대"]

# 메인 함수
if __name__ == "__main__":
    try:
        # 사용자로부터 입력 받기
        image_path = "C:/Users/lej55/Downloads/새 폴더/화장대.jpg"  # 첫 번째 인자: 이미지 경로
        model_path = "C:/Users/lej55/ssdamssdam/models/best_trained_model_v6.pth"  # 두 번째 인자: 모델 경로
        
        # 예측 수행
        pred_class, entropy = predict(image_path, model_path)
        
        # 예측된 품목명
        predicted_class_name = class_names[pred_class]
        
        # 결과 출력
        print(f"Predicted Class: {predicted_class_name}, Entropy: {entropy:.2f}")  # 품목명, 엔트로피 출력
    except Exception as e:
        print(f"Error: {str(e)}")

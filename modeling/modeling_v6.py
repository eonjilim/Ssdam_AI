import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 데이터셋 경로
data_dir = r'C:/Users/lej55/ssdamssdam/data/duplicate_data/merged_data_진열대'

# 전처리 및 데이터 증강 함수 정의
class CustomTransform:
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, img):
        # 기본적인 이미지 전처리 (크기 조정, 텐서 변환, 정규화)
        img = transforms.Resize((224, 224))(img)  # 이미지 크기 조정
        img = transforms.ToTensor()(img)  # 텐서로 변환
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)  # 정규화
        
        # 데이터 증강 (훈련 데이터에서만 적용)
        if self.augment:
            # 랜덤 좌우 반전
            if random.random() > 0.5:
                img = F.hflip(img)
            
            # 랜덤 회전
            angle = random.randint(-30, 30)
            img = F.rotate(img, angle)

            # 랜덤 밝기 조정
            brightness = random.uniform(0.8, 1.2)
            img = F.adjust_brightness(img, brightness)
            
            # 랜덤 크롭
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(224, 224))
            img = F.crop(img, i, j, h, w)
        
        return img


# 데이터셋 로딩 (전체 데이터 로딩)
original_transform = CustomTransform(augment=False)  # 증강 없이 전처리만 적용
dataset = datasets.ImageFolder(root=data_dir, transform=original_transform)

# 데이터셋을 훈련, 검증, 테스트 데이터로 분할
train_size = int(0.8 * len(dataset))  # 80% 훈련 데이터
val_size = int(0.1 * len(dataset))    # 10% 검증 데이터
test_size = len(dataset) - train_size - val_size  # 나머지 10% 테스트 데이터

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 사용자 지정: 증강 비율 설정 (0~1 범위)
augmentation_ratio = 0.3  # 예: 50%의 데이터를 증강

# 증강 데이터셋 생성 (훈련 데이터에서만)
augment_transform = CustomTransform(augment=True)
augmented_dataset = datasets.ImageFolder(root=data_dir, transform=augment_transform)

# 원본 훈련 데이터에 증강된 데이터 추가
num_augmented_samples = int(len(train_dataset) * augmentation_ratio)
indices = torch.randperm(len(augmented_dataset))[:num_augmented_samples]  # 무작위로 일부 데이터 선택
subset_augmented_dataset = torch.utils.data.Subset(augmented_dataset, indices)

# 원본 훈련 데이터셋과 증강 데이터셋을 병합
train_dataset = ConcatDataset([train_dataset, subset_augmented_dataset])

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 데이터셋 크기 확인
print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")
print(f"Augmented data percentage: {augmentation_ratio * 100}%")






import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score

# 모델 불러오기 (EfficientNet-B0)
model = timm.create_model('efficientnet_b0', pretrained=True)

# 마지막 Fully Connected Layer (FC) 수정 (Fine-tuning)
num_ftrs = model.classifier.in_features  # 기존 FC layer의 입력 차원
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # 첫 번째 FC layer
    nn.ReLU(),  # ReLU 활성화 함수
    nn.Dropout(0.5),  # Dropout 추가
    nn.Linear(512, len(dataset.classes))  # 클래스 개수만큼 출력
)

# 모델 파라미터 freeze (classifier 제외)
for param in model.parameters():
    param.requires_grad = False  # 모든 파라미터 freeze

# 'blocks'에서 마지막 5개 레이어만 학습하도록 설정
for param in model.blocks[-5:].parameters():
    param.requires_grad = True

# classifier 파라미터만 학습하도록 설정
for param in model.classifier.parameters():
    param.requires_grad = True

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # requires_grad가 True인 파라미터들만 optimizer에 전달
    lr=0.0001
)  # optimizer는 classifier와 일부 'blocks' 레이어만 사용
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 스케줄러

# 훈련 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # 훈련 단계
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 계산
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total

        print(f"Train Loss: {epoch_loss:.4f} \n \t Acc: {epoch_acc:.4f}")

        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 통계 계산
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / val_total

        print(f"Validation Loss: {val_loss:.4f} \n \t Acc: {val_acc:.4f}\n")

        # 학습률 스케줄러 변경 (검증 성능 개선시만 적용)
        # 모델이 개선되면 모델 가중치를 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            scheduler.step()  # 성능 개선이 있을 때만 학습률을 변경


    # 최종적으로 학습한 가장 좋은 모델 가중치 로드
    model.load_state_dict(best_model_wts)
    return model

# 모델 학습
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# 테스트 데이터셋 평가
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")

# 테스트 데이터 평가
evaluate_model(trained_model, test_loader)

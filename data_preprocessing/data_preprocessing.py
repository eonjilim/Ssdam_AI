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
data_dir = r'C:/Users/lej55/ssdamssdam/data/duplicate_data/merged_data'

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
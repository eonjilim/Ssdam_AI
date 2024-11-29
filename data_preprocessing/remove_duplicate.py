import os
from PIL import Image
import imagehash

# 중복된 이미지를 찾고 제거하는 함수
def remove_duplicate_images(image_folder):
    # 이미지 해시값을 저장할 딕셔너리
    image_hashes = {}
    duplicates = []

    # 폴더 내의 모든 이미지 파일을 확인
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # 이미지 파일만 처리
        if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                # 이미지 열기
                img = Image.open(image_path)
                # 이미지 해시값 계산 (perceptual hash 사용)
                hash_value = imagehash.phash(img)

                # 동일한 해시값이 존재하면 중복 이미지로 간주
                if hash_value in image_hashes:
                    duplicates.append(image_path)
                else:
                    image_hashes[hash_value] = image_path
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

    # 중복된 이미지 출력 및 삭제
    for duplicate in duplicates:
        print(f"Duplicate image found: {duplicate}")
        os.remove(duplicate)  # 중복 이미지 삭제

# 라벨 목록
labels = [
    'TV 받침', '거울', '빨래건조대', '서랍장', '쇼파', '시계', '안마의자', '의자', 
    '자전거', '장롱', '장식장', '진열대', '책꽂이', '책상', '책장', '침대', 
    '테이블', '피아노', '화장대'
]

# 각 라벨에 대해 중복 이미지 제거 실행
base_folder = "C:/Users/lej55/ssdamssdam/image_crawling/selenium_crawling/data"  # 기본 경로

for label in labels:
    label_folder = os.path.join(base_folder, label)  # 라벨 폴더 경로
    if os.path.exists(label_folder):
        print(f"Removing duplicates in folder: {label_folder}")
        remove_duplicate_images(label_folder)
    else:
        print(f"Folder for label '{label}' does not exist.")

import os
import shutil

# 소스 폴더와 대상 폴더 경로
source_folder1 = '/path/to/first_folder'  # 첫 번째 폴더 경로
source_folder2 = '/path/to/second_folder'  # 두 번째 폴더 경로
destination_folder = '/path/to/destination_folder'  # 합쳐진 데이터를 저장할 대상 폴더 경로

# 각 폴더 내의 레이블 폴더 목록을 가져오기
labels1 = os.listdir(source_folder1)
labels2 = os.listdir(source_folder2)

# 레이블별로 데이터를 합치기
for label in labels1:
    # 첫 번째 폴더에서 레이블별로 파일 복사
    label_folder1 = os.path.join(source_folder1, label)
    if os.path.isdir(label_folder1):
        label_folder_dest = os.path.join(destination_folder, label)
        if not os.path.exists(label_folder_dest):
            os.makedirs(label_folder_dest)
        for filename in os.listdir(label_folder1):
            src_file = os.path.join(label_folder1, filename)
            dest_file = os.path.join(label_folder_dest, filename)
            shutil.copy(src_file, dest_file)

for label in labels2:
    # 두 번째 폴더에서 레이블별로 파일 복사
    label_folder2 = os.path.join(source_folder2, label)
    if os.path.isdir(label_folder2):
        label_folder_dest = os.path.join(destination_folder, label)
        if not os.path.exists(label_folder_dest):
            os.makedirs(label_folder_dest)
        for filename in os.listdir(label_folder2):
            src_file = os.path.join(label_folder2, filename)
            dest_file = os.path.join(label_folder_dest, filename)
            shutil.copy(src_file, dest_file)

print("파일이 레이블별로 합쳐졌습니다.")

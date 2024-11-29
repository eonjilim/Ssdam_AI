import os
import sys
import urllib.request
import json

item = "장롱"

client_id = "20LpqdMBabcnhllH004Y"
client_secret = "LK4enA8gPh"
encText = urllib.parse.quote(item)

start = 1  # 검색 결과의 시작점 지정
url = f"https://openapi.naver.com/v1/search/image?display=100&start={start}&query={encText}"

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()

# 이미지 저장 경로
savePath = "C:/Users/lej55/OneDrive/바탕 화면/data/장롱"

if rescode == 200:
    response_body = response.read()
    result = json.loads(response_body)
    img_list = result['items']

    for i, img_list in enumerate(img_list, start):
        
        # 이미지 링크 확인
        print(img_list['link'])

        # 저장 파일명 및 경로
        FileName = os.path.join(savePath, item + str(i) + '.jpg')
        
        # 파일명 출력
        print('full name : {}'.format(FileName))
        
        # 이미지 URL이 유효한 형식인지 확인
        if img_list['link'].endswith(('.jpg', '.png', '.jpeg')):
            try:
                # 이미지 다운로드 URL 요청
                urllib.request.urlretrieve(img_list['link'], FileName)
                print(f"Downloaded: {FileName}")
            except Exception as e:
                print(f"Failed to download {img_list['link']}: {e}")
        else:
            print(f"Invalid image URL: {img_list['link']}")

    # 다운로드 완료 시 출력
    print("--------download succeeded--------")

else:
    print("Error Code:" + rescode)
from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import quote_plus
from urllib.request import urlopen
import os
import chromedriver_autoinstaller
import time


def save_images(browser, images, save_path, max_count=1000):
    count = 0  # 저장된 이미지 수
    index = 0  # 크롤링한 이미지 인덱스
    
    while count < max_count:  # 최대 1000개까지 저장
        try:
            if index >= len(images):  # 더 이상 가져올 이미지가 없으면 페이지를 스크롤하여 새로운 이미지 불러오기
                print("Scrolling to load more images...")
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 페이지 끝까지 스크롤
                time.sleep(7)  # 스크롤 후 잠시 대기하여 새로운 이미지 로딩

                images = browser.find_elements(By.CSS_SELECTOR, "div.thumb img")  # 다시 이미지 가져오기
                if not images:
                    print("No more images found!")
                    break  # 더 이상 이미지를 찾을 수 없으면 종료

                index = 0  # 스크롤 후 이미지를 처음부터 다시 처리하기 위해 인덱스를 초기화

            src = images[index].get_attribute('src')
            if src is None or not src.startswith("http"):
                print(f"Invalid image URL: {src}")
                index += 1
                continue  # 잘못된 URL일 경우 다음 이미지로

            t = urlopen(src).read()
            file_name = os.path.join(save_path, f"{count + 1}.jpg")
            with open(file_name, "wb") as file:
                file.write(t)
            print(f"Image saved: {file_name}")
            
            count += 1  # 이미지 저장 성공 시 count 증가

            # 요청 간 간격을 랜덤하게 설정 (정책 준수용)
            time.sleep(5 + (index % 3))  # 5-7초 기기

        except Exception as e:
            print(f"Failed to save image {count + 1}: {e}")
        
        index += 1  # 이미지 인덱스를 증가시키며 다음 이미지를 처리


def create_folder_if_not_exists(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print(f"Error creating directory {directory}: {e}")


def make_url(search_term):
    base_url = 'https://search.naver.com/search.naver?where=image&section=image&query='
    end_url = '&res_fr=0&res_to=0&sm=tab_opt&color=&ccl=2' \
              '&nso=so%3Ar%2Ca%3Aall%2Cp%3Aall&recent=0&datetype=0&startdate=0&enddate=0&gif=0&optStr=&nso_open=1'
    return base_url + quote_plus(search_term) + end_url


def crawl_images(search_term, max_count=1000):
    # URL 생성
    url = make_url(search_term)

    # ChromeDriver 설치 및 브라우저 열기
    chromedriver_autoinstaller.install()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 브라우저 창 띄우지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    browser = webdriver.Chrome(options=options)

    try:
        browser.implicitly_wait(3)
        browser.get(url)

        # 이미지 가져오기
        images = browser.find_elements(By.CSS_SELECTOR, "div.thumb img")
        if not images:
            print("No images found!")
            return

        # 저장 경로 설정
        save_path = "C:/Users/lej55/p_ssdam/seleniumcrawling/data/" + search_term  # 저장 경로 수정
        create_folder_if_not_exists(save_path)

        # 이미지 저장
        save_images(browser, images, save_path, max_count)

        print(f"Successfully saved {max_count} images for {search_term}! Path: {save_path}")
    except Exception as e:
        print(f"Error during crawling: {e}")
    finally:
        browser.quit()


if __name__ == '__main__':
    search_term = input('Enter search term: ')
    max_count = int(input('Enter number of images to save: '))
    crawl_images(search_term, max_count)
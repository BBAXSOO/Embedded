from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

pages = set()  # 중복 방지
url = input("url을 입력하세요: ")  # 원하는 url 입력받기

def crawl_page(url):  # 페이지를 크롤링
    response = requests.get(url)

    if response.status_code == 200:  # 성공
        print("웹 페이지를 성공적으로 불러왔습니다: ", url)
    else:
        print("웹 페이지 불러오기 실패: ", url)
        return

    # 객체 생성
    soup = BeautifulSoup(response.text, 'html.parser')

    # 페이지 제목 출력
    title = soup.title.string if soup.title else "제목없음"
    print("페이지 제목 :", title)

    # 뉴스 본문 출력 
    article_body = soup.select_one('.go_trans._article_content')

    if article_body:
        print("\n뉴스 본문:")
        print(article_body.get_text(strip=True))  # 본문 내용 출력
    else:
        print("뉴스 본문을 찾을 수 없습니다.")


# 처음 입력받은 URL을 크롤링
crawl_page(url)

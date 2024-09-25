import requests
from bs4 import BeautifulSoup

url = input("url를 입력하세요")


response = requests.get(url)

if response.status_code == 200:
    print("웹 페이지를 성공")

else:
    print("실패")


soup = BeautifulSoup(response.text, 'html.parser')

title = soup.title.string if soup.title else "제목없음"
print("페이지 제목 :", title)

news_headlines = soup.select('.news_tit')

if news_headlines:
    print("뉴스 헤드라인:")
    for headline in news_headlines:
        print(headline.get_text())

else:
    print("뉴스 헤드라인이 없습니다.")



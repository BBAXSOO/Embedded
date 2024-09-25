import requests
from bs4 import BeautifulSoup

url = 'https://www.naver.com'
response = requests.get(url)

if response.status_code == 200:
    print("웹 페이지를 성공")

else:
    print("실패")


soup = BeautifulSoup(response.text, 'html.parser')

title = soup.title.string
print("페이지 제목 :", title)

news_headlines = soup.select('.news_tit')

for headline in news_headlines:
    print(headline.get_text())
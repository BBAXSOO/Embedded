from bs4 import BeautifulSoup
from transformers import pipeline # type: ignore
import requests
import re
import datetime
from tqdm import tqdm # type: ignore 
import sys
import pandas as pd  # type: ignore

# 페이지 url 형식에 맞게 바꾸어 주는 함수
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    else:
        return num + 9 * (num - 1)

# 크롤링할 url 생성
def makeurl(search, start_pg, end_pg):
    urls = []
    for i in range(start_pg, end_pg + 1):
        page = makePgNum(i)
        url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&start={page}"
        urls.append(url)
    print("생성된 url 목록: ", urls)
    return urls

# html에서 속성을 뽑아온다
def news_attrs_crawler(articles, attrs):
    attrs_content = []
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content

# 네트워크 연결 문제 해결이라고 한다
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

# html에서 기사 URL을 추출하는 함수 (url 리스트에서 링크 반환)
def articles_crawler(url):
    # html 불러오기
    original_html = requests.get(url, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    # 뉴스 URL 추출
    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver, 'href')
    return url

# 검색어 입력
search = input("검색 키워드: ")
# 크롤링 시작 페이지 입력
page = int(input("\n크롤링 할 페이지 -> 숫자 : "))
print("\n크롤링할 시작 페이지: ", page, "페이지")
# 크롤링 종료 페이지 입력
page2 = int(input("\n크롤링 할 페이지 -> 숫자: "))
print("\n크롤링할 종료 페이지: ", page2, "페이지")

# 네이버 뉴스 URL 생성
urls = makeurl(search, page, page2)

# 뉴스 크롤러
news_titles = []
news_urls = []
news_contents = []
news_dates = []

for url in tqdm(urls):
    extracted_urls = articles_crawler(url)
    news_urls.append(extracted_urls)

# 리스트로 형태
def makeList(newlist, content):
    for i in content:
        for j in i:
            newlist.append(j)
    return newlist

news_urls_flat = []
makeList(news_urls_flat, news_urls)

# 네이버만 남기기
final_urls = []
for i in tqdm(range(len(news_urls_flat))):
    if "news.naver.com" in news_urls_flat[i]:
        final_urls.append(news_urls_flat[i])

# 뉴스 내용 크롤링
for i in tqdm(final_urls):
    news = requests.get(i, headers=headers)
    news_html = BeautifulSoup(news.text, "html.parser")

    # 제목
    title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
    if title is None:
        title = news_html.select_one("#content > div.end_ct > div > h2")
    
    # 본문
    content = news_html.select("article#dic_area")
    if not content:
        content = news_html.select("#articleBodyContents")
    
    # 내용
    content = ''.join(str(content))

    # html 태그를 제거 하는 거라고 한다
    pattern1 = '<[^>]*>'
    title = re.sub(pattern=pattern1, repl='', string=str(title))
    content = re.sub(pattern=pattern1, repl='', string=content)

    news_titles.append(title)
    news_contents.append(content)

    try:
        # 기사의 날짜를 가져온다고 한다
        html_date = news_html.select_one("div.media_end_head_info_datestamp > div > span")
        news_date = html_date.attrs['data-date-time']
        #attributeerror을 예외 처리 한다
    except AttributeError:
        news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
        news_date = re.sub(pattern=pattern1, repl='', string=str(news_date))

    news_dates.append(news_date)

# 중복 제거 후 기사 수 출력
print(f"\n검색된 기사 갯수: 총 {len(news_titles)}개")
print("\n[뉴스 제목]")
print(news_titles)
print("\n[뉴스 링크]")
print(final_urls)
print("\n[뉴스 내용]")
print(news_contents, "\n")


# 데이터프레임으로 전환한다
news_df = pd.DataFrame({'date': news_dates, 'title': news_titles, 'link': final_urls, 'content': news_contents})


# 중복 행 제거
news_df = news_df.drop_duplicates(keep='first', ignore_index=True)
print("중복 제거 후 행 개수: ", len(news_df))

# 내용을 파일 형태로 저장한다.
with open('news_contents_file.txt', 'w', encoding='utf-8') as f:
    for content in news_contents:
        f.write(content + "\n")
        

from langchain.document_loaders import TextLoader
documents = TextLoader("news_contents_file.txt", encoding='utf-8').load() # 파일 위치와 인코딩 지정
from langchain.text_splitter import RecursiveCharacterTextSplitter

#문서를 청크로 분할
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents) #docs 변수에 분할 문서를 저장

from langchain_openai import OpenAIEmbeddings
api_key="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"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)

#Chromdb에 백터 저장
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, api_key=api_key)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "해당 문서를 분석후 요약해줘"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)
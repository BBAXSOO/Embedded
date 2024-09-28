#install unstructured
#install sentence-transformers
#install chromadb
#install openai
#install langchain-openai

from langchain.document_loaders import TextLoader
documents = TextLoader("AI.txt").load() #파일 위치 지정

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
model_name = "gpt-4"
llm = ChatOpenAI(model_name=model_name, api_key=api_key)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "해당 문서의 내용을 최대한 요점만 짧게 요약해줘"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)

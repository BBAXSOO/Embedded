#pip install langchain
#pip install streamlit
#pip install openai

import streamlit as st
from langchain.chat_models import ChatOpenAI
st.set_page_config(page_title="뭐든지 질문하세요~")
st.title("질문을 입력해주세요")

import os
os.environ["OPENAI_API_KEY"] = "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"

def generate_response(input_text):
    llm = ChatOpenAI(temperature=0,
                     model_name='gpt-4',
                    )
    st.info(llm.predict(input_text))

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('보내기')
    generate_response(text)

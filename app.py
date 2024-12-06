import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from langchain_groq import ChatGroq

os.environ["langchain_api_key"] = os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "simple Qand A chatbot"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Respond to the user's question."),
    HumanMessagePromptTemplate.from_template("questions: {questions}")
])

def generate_response(questions, engine, temperature, max_tokens):
    model = ChatGroq(model=engine, temperature=temperature, max_tokens=max_tokens)
    output = StrOutputParser()
    chain = prompt | model | output
    answer = chain.invoke({"questions": questions})
    return answer

st.title("Q & A chatbot")
engine = "gemma2-9b-it"
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=1, max_value=1000, value=150)

user = st.text_input("Please enter what you want to know")

if user:
    try:
        response = generate_response(user, engine, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please enter text.")

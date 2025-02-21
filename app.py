import os
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# langsmith tracing
os.environ["LANGCHAIN_API_KEY"]=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Groq Models"

# prompt template
prompt=ChatPromptTemplate.from_messages([
  ("system","You are a helpful assistant, Please response to the user query."),
  ("user","Question:{question}")
])

def generate_response(question,api_key,llm,temperature,max_token):
  llm=ChatGroq(model=llm,groq_api_key=api_key,max_tokens=max_token,temperature=temperature)
  parser=StrOutputParser()
  chain=prompt|llm|parser
  answer=chain.invoke({'question':question})
  print(answer)
  return answer

# streamlit frontend
st.title("Q&A Chatbot with Groq Models")

# sidebar
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter OpenAI API Key:",type="password")

# drop down for model selection
llm=st.sidebar.selectbox("Select a Groq Model",['deepseek-r1-distill-llama-70b','gemma2-9b-it','llama3-8b-8192','mixtral-8x7b-32768','whisper-large-v3-turbo'])

# response parameter adjustment
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# main
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
  response=generate_response(user_input,api_key,llm,temperature,max_tokens)
  st.write(response)
else:
  st.write("Please ask the question!")

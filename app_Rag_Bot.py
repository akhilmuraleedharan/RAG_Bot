__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import sqlite3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai  
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
st.title("Rag Q/A Bot for Pdf files")

st.sidebar.text("This is an RAG Q/A Bot to answer questions related to uploaded document in the pdf format!!!")
st.sidebar.text("You can upload the document and ask the bot questions based on the document after clicking the Browse files button")
st.sidebar.text("Example Queries include:")
st.sidebar.text(" 1.Provide a summary of the uploaded document")
st.sidebar.text(" 2.What are some technical terms used in the document")
uploaded_file = st.sidebar.file_uploader("File upload", type=["pdf"], accept_multiple_files=False)
if uploaded_file :
    
    
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name

    loader =  PyPDFLoader('temp.pdf')
    data =  loader.load()

    textsplitter =  RecursiveCharacterTextSplitter(chunk_size = 1000)
    docs =  textsplitter.split_documents(data)


    store_vector = Chroma.from_documents(documents=docs, embedding= GoogleGenerativeAIEmbeddings(model='models/embedding-001'),persist_directory="./data")
    
    retriver = store_vector.as_retriever(search_type =  "similarity", search_kwargs = {"k":10})

    query = st.chat_input("Say Something")
    prompt = query
    llm =  ChatGoogleGenerativeAI(model =  "gemini-1.5-flash", temperature=0.3,max_tokens=500)

    system_prompt = (
        "You are an assistant for question answer tasks"
        "Use the following pieces of retrived context to answer"
        "the question.If you don't know the answer, say that you"
        "dont know. Use four sentences maximum  and keep the"
        "answer concise."
        "\n\n"
        "{context}"
        
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("human", "{input}"),
        ]
    )

    if query:
        question_answer_chain = create_stuff_documents_chain(llm,prompt)
        rag_chain =  create_retrieval_chain(retriver,question_answer_chain)
        response = rag_chain.invoke({"input": query})
        st.write(response["answer"])        
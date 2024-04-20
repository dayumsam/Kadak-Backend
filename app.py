import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from flask import Flask


from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

# get envirionment 
os.environ.get("OPENAI_API_KEY")

# pip install langchain langchain_openai pypdf faiss-gpu pydantic==1.10.8

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
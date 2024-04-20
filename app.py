import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from flask import *
from flask import request
from fileinput import filename
import requests
import json

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from werkzeug.utils import secure_filename

from langchain.agents import AgentType, initialize_agent

# get envirionment 
os.environ.get("OPENAI_API_KEY")

# pip install langchain langchain_openai pypdf faiss-gpu pydantic==1.10.8

app = Flask(__name__)

UPLOAD_FOLDER = './pdfs/'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/injest', methods=['GET', 'POST'])
# def injest_pdf():
#     if request.method == 'POST':
#         print("Post req")
#         file = request.files.get('file')
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             print("yes")
#         else:
#             print("oops")
#     return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

@app.route('/injest', methods=['GET', 'POST'])
def injest_pdf():
    if request.method == 'POST':
        if os.path.exists('./pdfs/data.pdf'):
            return json.dumps({'success':False}), 300, {'ContentType':'application/json'}
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        #return render_template("acknowledge.html", name = file.filename)
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    class DocumentInput(BaseModel):
        question: str = Field()

    loader = PyPDFLoader("./pdfs/data.pdf")
    pages = loader.load_and_split()

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    tools = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = text_splitter.split_documents(pages)

    retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever()

    tools.append(
        Tool(
            args_schema=DocumentInput,
            name="transformer",
            description=f"useful when you want to answer questions about a given university catalog",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )

    #print(docs)

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0613",
    )

    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    print(agent({"input": "List the courses that require prerequisites."}))

    #return render_template("front.html")
    return "<p>Hello World</p>"
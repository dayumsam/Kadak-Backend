import os
from dotenv import load_dotenv, find_dotenv

from flask import *
from fileinput import filename
import json

from langchain.agents import Tool
from langchain.tools.base import StructuredTool

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# libraries for data extraction
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Chroma
from langchain_chroma import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

from langchain.agents import AgentType, initialize_agent,  AgentExecutor, create_openai_tools_agent

load_dotenv(find_dotenv())

# Database
# import sqlite3
# con = sqlite3.connect("kadak.db", check_same_thread=False)
# cur = con.cursor()

# Create DB if not exists
# cur.execute("CREATE TABLE IF NOT EXISTS documents(_id integer primary key autoincrement, title varchar(100) not null, data text)")

# get envirionment 
os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)

# PDF file injesting workflow

UPLOAD_FOLDER = './pdfs/'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Check if UPLOAD_FOLDER does not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ContactInfo(BaseModel):
    phone: Optional[str] = Field(default=None, description="Phone number for contact")
    email: Optional[str] = Field(default=None, description="Email address for contact")
    address: Optional[str] = Field(default=None, description="Physical address of the location")
    website: Optional[str] = Field(default=None, description="Website URL")

class Break(BaseModel):
    breakName: str = Field(..., description="Name of the break period")
    startDate: str = Field(..., description="Start date of the break")
    endDate: str = Field(..., description="End date of the break")

class ImportantDate(BaseModel):
    description: str = Field(..., description="Description of the important date")
    date: str = Field(..., description="Specific date of the important date event")

class AcademicTerm(BaseModel):
    term: str = Field(..., description="Academic term, e.g., Fall 2018")
    startDate: str = Field(..., description="Start date of the academic term")
    endDate: str = Field(..., description="End date of the academic term")
    breaks: List[Break] = Field(..., description="List of breaks during the term")
    importantDates: List[ImportantDate] = Field(..., description="List of important dates within the term")

class Instructor(BaseModel):
    name: str = Field(..., description="Name of the instructor")
    email: str = Field(..., description="Email address of the instructor")
    officeHours: str = Field(..., description="Scheduled office hours for the instructor")
    officeLocation: str = Field(..., description="Office location of the instructor")

class Course(BaseModel):
    title: str = Field(..., description="Title of the course")
    code: str = Field(..., description="Code identifying the course")
    description: str = Field(..., description="Detailed description of the course")
    creditHours: int = Field(..., description="Number of credit hours for the course")
    department: str = Field(..., description="Department offering the course")
    prerequisites: List[str] = Field(..., description="List of prerequisite courses")
    corequisites: List[str] = Field(..., description="List of corequisite courses")
    offered: List[str] = Field(..., description="Terms during which the course is offered")
    instructor: Instructor = Field(..., description="Instructor of the course")
    syllabusURL: Optional[str] = Field(default=None, description="URL to the course syllabus")
    assessmentMethods: List[str] = Field(..., description="Methods of assessment used in the course")

class Requirement(BaseModel):
    name: str = Field(..., description="Name of the degree requirement")
    credits: int = Field(..., description="Credit hours required for the requirement")
    mandatory: bool = Field(..., description="Whether the requirement is mandatory")

class DegreeProgram(BaseModel):
    degreeName: str = Field(..., description="Name of the degree program")
    degreeType: str = Field(..., description="Type of degree, e.g., Bachelor, Master")
    school: str = Field(..., description="School offering the degree")
    requiredCredits: int = Field(..., description="Total credit hours required for the degree")
    coreCourses: List[str] = Field(..., description="List of core courses for the degree")
    electiveCourses: List[str] = Field(..., description="List of elective courses for the degree")
    requirements: List[Requirement] = Field(..., description="Specific requirements for the degree")
    graduationRequirements: List[str] = Field(..., description="General graduation requirements")

class GradingScale(BaseModel):
    grade: str = Field(..., description="Letter grade")
    range: str = Field(..., description="Percentage range for the grade")

class AcademicPolicies(BaseModel):
    gradingScale: List[GradingScale] = Field(..., description="Grading scale used")
    attendancePolicy: str = Field(..., description="Policy on class attendance")
    transferCreditsPolicy: Dict[str, str] = Field(..., description="Policy on transferring credits from other institutions")
    academicStanding: List[Dict[str, str]] = Field(..., description="Standards for academic standing")
    examinationPolicies: str = Field(..., description="Policies regarding examinations")

class Event(BaseModel):
    title: str = Field(..., description="Title of the event")
    description: str = Field(..., description="Description of the event")
    date: str = Field(..., description="Date of the event")
    location: Optional[str] = Field(default=None, description="Location of the event if applicable")

class JsonOutput(BaseModel):
    """Schema for json output to be generated from the given university catalog"""
    UniversityInfo: str = Field(default=None, description="Information about the university")
    AcademicCalendar: List[AcademicTerm] = Field(..., description="Academic calendar details")
    Courses: List[Course] = Field(..., description="List of courses offered")
    DegreePrograms: List[DegreeProgram] = Field(..., description="List of degree programs available")
    AcademicPolicies:str = Field(default=None, description="Academic policies of the institution")
    Events: str = Field(default=None, description="List of events at the institution")

# Vectorizing workflow
# TODO: only hit if file does not exist
def loadPdfPages(filename):
    print("Loading pdf....")

    loader = PyPDFLoader(f"./pdfs/{filename}")
    pages = loader.load_and_split()

    # TODO: put key in a db
    # cur.execute("INSERT INTO documents(title) VALUES (?)", (str(filename),))
    # print("Added Entry to the ledger")

    print("Done!")
    return pages

def textChunking(pages):
    print("Splitting into chunks....")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=10,
    )
    docs = text_splitter.split_documents(pages)

    print("Done!")
    return docs

def vectorStoreGenerator(docs, filename):
    print("Now generating vectorstore....")

    db = FAISS.from_documents(docs, OpenAIEmbeddings())

    print("Done!")
    return db

# https://python.langchain.com/docs/integrations/vectorstores/chroma/#basic-example-including-saving-to-disk

@app.route('/injest', methods=['POST'])
def injest_pdf():
    if request.method == 'POST':
        file = request.files['file']

        if os.path.exists(f'{UPLOAD_FOLDER}{file.filename}'):
            print("File already processed, using existing vector store")
            # currentDB = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
        #     return json.dumps({'success':False}), 300, {'ContentType':'application/json'}
        else:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Start the RAG workflow

        # TODO: check if collection exists
            # res = cur.execute("SELECT * FROM documents WHERE title=(?)", (file.filename,))
            # print(cur.fetchall(), file.filename)
            # if(len(cur.fetchall()) < 1):

            pages = loadPdfPages(file.filename)
            docs = textChunking(pages)
            currentDB = vectorStoreGenerator(docs, file.filename)

        # currentDB = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

        retriever = currentDB.as_retriever()
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview",  model_kwargs={"response_format": {"type": "json_object"}})


        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # response = retrieval_chain.invoke({"input": "Generate a json format output with all the important data from the document make sure to capture as much data as possible and keep it clean in a JSON format make it as long as possible"})

        # print(response["answer"])

        tools = []

        catalog = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        tools.append(
            Tool(
                args_schema=JsonOutput,
                name="catalog_reader",
                description="useful for when you need to answer questions about university from the university catalog Input should be a fully formed question.",
                func=StructuredTool.from_function(catalog.run)
            )
        )

        # Get the prompt to use - you can modify this!
        # prompt = hub.pull("hwchase17/openai-tools-agent")
        # agent = create_openai_tools_agent(llm, tools, prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # print(agent_executor.invoke({"input": "Generate a json format output with important data about the college use data from the document"}))

        agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=tools,
            llm=llm,
            verbose=True,
            handle_parsing_errors=True
        )

        print(agent({"input": "Generate a json format output with all the important data about the college use data from the document"}))

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
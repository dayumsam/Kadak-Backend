from flask import Flask

# pip install langchain langchain_openai pypdf faiss-gpu pydantic==1.10.8

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
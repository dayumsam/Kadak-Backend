import os

from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

load_dotenv()  # Load environment variables from a .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def load_pdf_pages(file_path):
    # Load the PDF and split into chunks
    text_content = []
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())
    return text_content

def process_data_with_ai(text_content):
    # Process the text with AI
    try:
        response = client.chat.completions.create(
            model="text-davinci-003",
            prompt="You are a text parser and you have to parse the following text into a structured json with all the important data.:\n\n" + " ".join(text_content),
            response_format={ "type": "json_object" }
        )

        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Failed to process text with AI: {str(e)}")
        return None

process_data_with_ai(load_pdf_pages("pdfs/data.pdf"))
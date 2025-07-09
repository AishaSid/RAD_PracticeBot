import os
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# for vector store and retrieval
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import RetrievalQA


# Load env variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_key"))

#  Direct Gemini usage for response
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content("Explain the RAD model in 1 sentence only.")
print("Gemini says:", response.text)

#  Load PDF (dummy file for testing)
loader = PyPDFLoader("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
pages = loader.load_and_split()
print(len(pages), "pages loaded from the PDF.")

#  Convert pages into embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_key")
)

#  Create vector store
vector = FAISS.from_documents(pages, embeddings)
print("Vector store created.")

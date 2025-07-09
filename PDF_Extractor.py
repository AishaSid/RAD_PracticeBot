import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# for vector store and retrieval
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # Needed for prompt

# Page configuration
st.set_page_config(
    page_title="PDF Extraction Bot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, clean styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
# Header section
st.markdown("""
<div class="main-container">
    <h1 class="main-title">Task Requirement Extraction Bot</h1>
    <h2 class="main-subtitle">A simple RAD application</h2>
    <p class="subtitle">Upload your project documents and extract key requirements with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)



uploaded_file = st.file_uploader(
    "Choose a PDF file containing project requirements", 
    type=["pdf"], 
    accept_multiple_files=False,
    help="Upload a PDF document containing project details, requirements, or technical specifications"
)

# user query for RAG 
user_query = st.text_area("Ask something about the document", value="Summarize the project requirements.")

# Submit button
submit = st.button("Extract Requirements", use_container_width=True)

# Helper to ensure Google API key is set
def ensure_google_key():
    google_key = os.getenv("GOOGLE_key")
    if not google_key:
        st.markdown('<div class="error-message">Google API Key Required</div>', unsafe_allow_html=True)
        google_key = st.text_input("Enter your Google API key:", type="password")
        if not google_key:
            st.warning("Please enter your Google API key to proceed.")
            st.stop()
        os.environ["GOOGLE_key"] = google_key
    return google_key

# Main RAG code
def create_docs(uploaded_file, google_key):
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_key
    )
    vectorstore = FAISS.from_documents(pages, embeddings)

    # Prompt template for the LLM
    prompt_template = """
You are an intelligent assistant helping to understand project documents.

Use the following document context to answer the question below.

Context:
{context}

Question:
{question}

Please extract and summarize the following:

1. What is the main goal or purpose of the project?
2. What tasks or steps are required to complete the project?
3. Which tools, frameworks, or technologies are mentioned or expected to be used?
4. What are the expected deliverables or output files?
5. Mention any timelines, milestones, or submission details if available.

Your response should be in this format:

### Answer to your Question:
 [your response here.. briefly in 2-3 lines]

### Summary of the Document:
- [One short paragraph]

### Requirements:
1. [List each requirement]
2. ...

### Frameworks / Tools:
- [List of tools or frameworks]

### Expected Deliverables:
- [List of files or output expected]

### Additional Notes:
- [Any deadlines, guidelines, or instructions found in the PDF]

Be clear, concise, and structured in your output.
"""
    # Use PromptTemplate with 'context' as input variable
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        max_output_tokens=1000,
        top_p=0.95,
        top_k=40,
        google_api_key=google_key
    )

    retrv = vectorstore.as_retriever()
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # Set up RetrievalQA chain
    retv_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retrv,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    
    resp = retv_chain.invoke({"query": user_query})
    ans = resp.get("result", "No answer found.")
    print("Response:", ans)
    print("--------------------------------DONE-----------------------------------")

    # Clean up temp file
    os.remove("temp.pdf")

    return ans

# Main logic
if submit:
    if not uploaded_file:
        st.markdown('<div class="error-message">Please upload a PDF file first</div>', unsafe_allow_html=True)
        st.stop()

    with st.spinner("Processing your document... This may take a moment"):
        try:
            google_key = ensure_google_key()
            st.markdown('<div class="success-message">Document uploaded successfully! Analyzing content...</div>', unsafe_allow_html=True)
            
            response = create_docs(uploaded_file, google_key)
            
            # Display response in a styled container
            st.markdown("""
            <div class="response-container">
                <h3 class="response-title">Extracted Requirements</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Render the response with preserved Markdown and custom CSS
            st.markdown(
                f"""
                <div class="response-markdown">
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                response,
                unsafe_allow_html=True  # Allow HTML for custom classes, but Markdown will render headings/numbering
            )
            # Add custom CSS to style the markdown output
            st.markdown("""
            <style>
            .response-markdown h3, .response-markdown h4, .response-markdown h5, .response-markdown h6 {
                color: #2d5be3;
                margin-top: 1.2em;
                margin-bottom: 0.5em;
            }
            .response-markdown ul, .response-markdown ol {
                margin-left: 1.5em;
                margin-bottom: 1em;
            }
            .response-markdown li {
                margin-bottom: 0.3em;
                font-size: 1.05em;
            }
            .response-markdown p {
                margin-bottom: 0.8em;
                font-size: 1.08em;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Success message
            st.markdown('<div class="success-message">Analysis completed successfully!</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-message">Error processing document: {str(e)}</div>', unsafe_allow_html=True)


# Footer
st.markdown("""
<div class="footer">
    <p>Built with Streamlit and Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)
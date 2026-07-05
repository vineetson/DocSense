import os
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()
# Set your Google API key as an environment variable or directly
# It's generally recommended to use environment variables for sensitive keys
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
# If you're setting it directly in the code for testing, be cautious in production
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
st.header("DocSense Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract contents from the PDF and chunk it
if file is not None:
    # extract the text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    # st.write(text) # You can uncomment this for debugging if needed

    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks) # This should now display the chunks

    # generating embeddings using GoogleGenerativeAIEmbeddings
    # Use a known working model like "gemini-embedding-2-preview"
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=GOOGLE_API_KEY)
        # Or simply: embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY)
        # Or for a newer model if available: embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-1.5-flash", google_api_key=GOOGLE_API_KEY)

        #store embeddings in vector db
        vector_store = FAISS.from_texts(chunks, embeddings)

        st.success("PDF processed and embeddings generated!")

    except Exception as e:
        st.error(f"An error occurred during embedding generation: {e}")
        st.warning("Please check your Google API Key and the embedding model name.")
        st.info("Try using 'text-embedding-004' as the model, or ensure your API key is valid and has access to Generative AI services.")

    #get user question 
    user_question=st.text_input("Type your question here")

    # generate anser
    # question -> embeddings -> similarity search -> results to LLM -> response (CHAIN)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    # define the LLM and prompts
    # llm=GoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GOOGLE_API_KEY)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3,  # Gemini 3.0+ defaults to 1.0, temperature denotes randomness, lower means more deterministic, higher means more creative
        max_tokens=1000,
        google_api_key=GOOGLE_API_KEY
    )

    # provide the prompts
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response=chain.invoke(user_question)
        st.write(response)
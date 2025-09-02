"""Simple RAG-based chatbot using LangChain, Chroma, and Azure OpenAI.

Usage:
    python rag_chatbot.py --rebuild    # Rebuilds the vector store from docs/
    python rag_chatbot.py --query "Your question"    # Ask a question

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    AZURE_OPENAI_CHAT_DEPLOYMENT
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "docs"
DB_PATH = BASE_DIR / "chroma_db"


def build_vector_store() -> None:
    """Load documents and persist a Chroma vector store."""
    loader = DirectoryLoader(str(DOCS_PATH), glob="*.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version="2023-12-01-preview",
    )

    Chroma.from_documents(splits, embeddings, persist_directory=str(DB_PATH))


def get_chain() -> RetrievalQA:
    """Create a RetrievalQA chain backed by Chroma and Azure OpenAI."""
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version="2023-12-01-preview",
    )
    vectordb = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version="2023-12-01-preview",
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    )
    return RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple RAG chatbot")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store")
    parser.add_argument("--query", type=str, help="Question for the chatbot")
    args = parser.parse_args()

    load_dotenv()

    if args.rebuild:
        build_vector_store()

    if args.query:
        chain = get_chain()
        print(chain.run(args.query))


if __name__ == "__main__":
    main()

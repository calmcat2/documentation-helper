from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import asyncio

load_dotenv()


async def ingest_doc():
    print("Loading documents...")
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_doc = loader.load()
    print(f"Loaded {len(raw_doc)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    text = text_splitter.split_documents(raw_doc)
    print(f"Split into {len(text)} chunks")
    print("Updating document URLs...")
    for doc in text:
        url = doc.metadata["source"]
        url = url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": url})

    vectorstore = PineconeVectorStore(
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"
        ),
        index_name="langchain-doc-index",
    )
    batch_size = 100
    for i in range(0, len(text), batch_size):
        batch = text[i : i + batch_size]
        print(f"Ingesting batch {i // batch_size + 1} of {len(text) // batch_size + 1}")
        await vectorstore.aadd_documents(batch)
    print("Ingestion complete. Documents added to Pinecone vector store.")


if __name__ == "__main__":
    asyncio.run(ingest_doc())


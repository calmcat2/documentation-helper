from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from typing import List
from langchain_core.messages import BaseMessage


load_dotenv()


def docsearch(query: str, chat_history: List[BaseMessage] = []):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07"
    )
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name="langchain-doc-index",
    )

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        verbose=True,
        temperature=0
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retriever_with_memory = create_history_aware_retriever(
        llm=llm, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    chain = create_retrieval_chain(
        retriever=retriever_with_memory,
        combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"], result["context"]


if __name__ == "__main__":
    print(docsearch("Why use LangChain?"))

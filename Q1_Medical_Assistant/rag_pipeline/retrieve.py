from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from load_db import vector_store

from dotenv import load_dotenv

load_dotenv()

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a RetrievalQA chain that uses the retriever and LLM
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Test the retrieval without re-embedding
query = input("Question: ")
answer = qa_chain.invoke(query)

print("\nAnswer:", answer["result"])

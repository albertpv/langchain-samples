import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = "gpt-3.5-turbo"
print("LLM", llm_name)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print("vector db size" , vectordb._collection.count())
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)
print("question", question)
print("topics vectordb:", docs)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain({"query": question})
print("question=>", question)
print("answer RetrievalQA=>", result["result"])

from langchain.prompts import PromptTemplate
print("---now test adding a prompt template---")
# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

question = "Is probability a class topic?"
print("question=>",question)

result = qa_chain({"query": question})

print("answer=>",result["result"])
result["source_documents"][0]

print("---chainMR with map reduce--")
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
print("question chainMR=>",question)
print("answer=>",result["result"])

#Get key from https://smith.langchain.com/
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
langchain_api_key = os.environ['LANGCHAIN_API_KEY']
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]
print("question=>",question)
print("answer=>",result["result"])

print("---chainMR with refine--")
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]
print("question=>",question)
print("answer=>",result["result"])

### RetrievalQA limitations: No conversational history
print("Retrieval QA Limitations: no chained questions")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]
print("question=>",question)
print("answer=>",result["result"])

question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
result["result"]
print("question=>",question)
print("answer=>",result["result"])

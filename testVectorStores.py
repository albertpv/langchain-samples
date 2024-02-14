import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
print("number of splits:", len(splits))

# Test embeddings
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"
print("sentence 1:",sentence1)
print("sentence 2:",sentence2)
print("sentence 3:",sentence3)

try:
    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)

    import numpy as np
    print("---")
    print("Similarity sentence1 and sentence 2: ", np.dot(embedding1, embedding2))
    print("Similarity sentence1 and sentence 3: ", np.dot(embedding1, embedding3))
    print("Similarity sentence2 and sentence 2: ", np.dot(embedding2, embedding2))
except Exception as e:
    print("Something went wrong in embedding=>", str(e))

from langchain_community.vectorstores import Chroma
persist_directory = 'docs/chroma/'

question = "is there an email i can ask for help"
print("Question: ", question)
try:
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print(vectordb._collection.count())
    docs = vectordb.similarity_search(question,k=3)
    print("Length of answer:",len(docs))
    print("Page content: ", docs[0].page_content)
    vectordb.persist()
except Exception as e:
    print("Something went wrong in Chroma =>", str(e))

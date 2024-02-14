import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.vectorstores import Chroma
persist_directory = 'docs/chroma/'
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
vectordb = Chroma( persist_directory=persist_directory,embedding_function=embedding)
print("vector db count" , vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]
smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"

print("texts:", texts)
print("question:",question)
print("answer similarity",smalldb.similarity_search(question, k=2))
print("answer max marginal relevance", smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))


question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)

### Addressing Diversity: Maximum marginal relevance
#Last class we introduced one problem: how to enforce diversity in the search results.
#`Maximum marginal relevance` strives to achieve both relevance to the query *and diversity* among the results.
print("--Addressing diversity--")

print("MS search first item:", docs_ss[0].page_content[:100])
print("MS search second item:",docs_ss[1].page_content[:100])
print("Oh , same item two times :-(  Let's attempt MMR (maximum marginal relevance search)")
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
docs_mmr[0].page_content[:100]

print("MMR search first item:", docs_mmr[0].page_content[:100])
print("MMR search second item:",docs_mmr[1].page_content[:100])
print("Now the two items are different! :-)")

### Addressing Specificity: working with metadata
#In last lecture, we showed that a question about the third lecture can include results from other lectures as well.
#
#To address this, many vectorstores support operations on `metadata`.
# `metadata` provides context for each embedded chunk.
print("--Addressing specificity working with metadata using self-query retriever--")
print("question", question)
question = "what did they say about regression in the third lecture?"

print("approach 1: fix metadata manually (just add a filter)")
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)
for d in docs:
    print(d.metadata)

print("approach 2: Let that filter be added by a language model")
from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]
#Note: The default model for OpenAI ("from langchain.llms import OpenAI") is text-davinci-003. Due to the deprication of OpenAI's model text-davinci-003 on 4 January 2024, you'll be using OpenAI's recommended replacement model gpt-3.5-turbo-instruct instead.
document_content_description = "Lecture notes"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
docs = retriever.get_relevant_documents(question)
print(docs)
for d in docs:
    print(d.metadata)

### Additional tricks: compression
#Another approach for improving the quality of retrieved docs is compression.
#Information most relevant to a query may be buried in a document with a lot of irrelevant text.
#Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
#Contextual compression is meant to fix this.
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
question = "what did they say about matlab?"
print(" compression retriever output =>")

compressed_docs = compression_retriever.get_relevant_documents(question) #Under the hood uses semantic search (duplicates are possible)
pretty_print_docs(compressed_docs)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="mmr")
)
print("MMR and compression retriever output =>")
question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

## Other types of retrieval (no vector database)
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)
# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)


question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
print("question:",question,"--","answer SVM",docs_svm[0])
print("question:",question,"--","answer TFIDF",docs_tfidf[0])

question = "what did they say about matlab?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
print("question:",question,"--","answer SVM",docs_svm[0])
print("question:",question,"--","answer TFIDF",docs_tfidf[0])

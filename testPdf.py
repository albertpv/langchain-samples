import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
print("len",len(pages))
page = pages[0]
print("content",page.page_content[0:500])
print("metadata",page.metadata)

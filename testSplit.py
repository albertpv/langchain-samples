import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
text1 = 'abcdefghijklmnopqrstuvwxyz'
print("r_splitter",r_splitter.split_text(text1))

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print("r_splitter", r_splitter.split_text(text2))

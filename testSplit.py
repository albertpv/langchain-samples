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
print(text1)
print("----recursive splitter chunk size 26 chunk overlap 4------")
print("r_splitter",r_splitter.split_text(text1), "=> cannot split")


text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print(text2)
print("----- recursive splitter----")
print("r_splitter", r_splitter.split_text(text2),"=> can split now")


text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print(text3)
print("--splitter test (recursive character text splitter and character text splitter)-------")
print("r_splitter",r_splitter.split_text(text3))
print("c_splitter",c_splitter.split_text(text3))


some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

print ("---process text and test splitters with chunk sizse:450 chunk overlap:0------")
print(some_text)

print("text length", len(some_text))
print("----splitter test------")
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

print("c_splitter",c_splitter.split_text(some_text))
print("r_splitter",r_splitter.split_text(some_text))

print("----splitter test with chunk size 150----")
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)

print("r_splitter",r_splitter.split_text(some_text))
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
print("r_splitter size 150 with more separators" ,r_splitter.split_text(some_text))

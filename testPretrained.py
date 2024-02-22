#Based in https://github.com/curiousily/Get-Things-Done-with-Prompt-Engineering-and-LangChain/blob/master/13.chat-with-multiple-pdfs-using-llama-2-and-langchain.ipynb
import torch
from auto_gptq import AutoGPTQForCausalLM,BaseQuantizeConfig
from langchain_core.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
meta_images = convert_from_path("./docs/misc/consellescolar2.pdf", dpi=88)
loader = PyPDFDirectoryLoader("docs/misc")
docs = loader.load()
DEVICE = "cpu"
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
len(texts)
db = Chroma.from_documents(texts, embeddings, persist_directory="db")
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"   #"TheBloke/Llama-2-13B-chat-GPTQ"

model_basename = "model"
print("before tokenizer")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print("after tokenizer")
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=quantize_config
)
print("after create model")
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

print("after prompt")
def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
print("after HuggingFacePipeline")
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
question="Qui és el president del consell escolar?"
result = qa_chain(question)
print("question=>",question)
print("answer=>",result["source_documents"][0].page_content)

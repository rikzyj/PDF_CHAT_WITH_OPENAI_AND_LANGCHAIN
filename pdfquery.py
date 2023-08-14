import textwrap

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# API key from openai
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# location of the pdf file.
reader = PdfReader(input("Paste the file path of the PDF you would like to chat with :"))

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# split the text into smaller chunks to comply with OpenAI's token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# print(len(texts))

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

while True:
    query = input("Enter your prompt or type \"exit\" to exit chat? \n")
    if query=="exit":
        break
    else:
        docs = docsearch.similarity_search(query)
        output = chain.run(input_documents=docs, question=query)
        # Wrap the text to a specified width (e.g., 80 characters) and print it
        wrapped_text = textwrap.fill(output, width=80)
        print(wrapped_text)

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant hepling with explaining machine learning techniques"),
    ("user", "{input}")
])

llm=Ollama(model="llama2")
chain = prompt | llm | output_parser

message=chain.invoke("write a python implementation of knn")

print(message)
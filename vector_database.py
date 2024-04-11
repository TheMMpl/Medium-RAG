from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import os.path



class VectorDatabase:
    def __init__(self,mode):
        if mode=="local":
            self.embeddings=OllamaEmbeddings()
            self.splitter= RecursiveCharacterTextSplitter()
        else:
            self.embeddings= OpenAIEmbeddings()
            self.splitter= RecursiveCharacterTextSplitter()

        if os.path.exists("faiss_index"):
            self.load_db()
        else:
            self.create_db()
    
    def prepare_data(self):
        loader = CSVLoader(
        file_path="medium.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["Title", "Text"],
        },
        )
        return loader.load()
    
    def create_db(self):
        docs=self.prepare_data()
        documents = self.splitter.split_documents(docs[1:])
        self.vector = FAISS.from_documents(documents, self.embeddings)
        self.vector.save_local("faiss_index")
    
    def load_db(self):
        self.vector=FAISS.load_local("faiss_index", self.embeddings,allow_dangerous_deserialization=True)

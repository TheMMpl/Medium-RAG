from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import os.path



class VectorDatabase:
    def __init__(self,mode,question_type):

        if question_type=="specific":
            self.chunk_size=200
            self.db_path="db_index_specific"
        else:
            self.chunk_size=1000
            self.db_path="db_index"

        self.splitter= RecursiveCharacterTextSplitter(chunk_size= self.chunk_size,chunk_overlap= 200,separators=['\nclass ', '\ndef ', '\n\tdef ',"\n\n", "\n", " ", ""])
        self.parent_splitter= RecursiveCharacterTextSplitter(chunk_size= self.chunk_size*8,chunk_overlap= 200,separators=['\nclass ', '\ndef ', '\n\tdef ',"\n\n", "\n", " ", ""])

        if mode=="local":
            self.embeddings=OllamaEmbeddings()  
        else:
            self.embeddings= OpenAIEmbeddings()
        
        self.docs=self.prepare_data()

        if os.path.exists(self.db_path):
            self.load_db()
        else:
            self.create_db()
        
        self.store=InMemoryStore()
        self.retriever = ParentDocumentRetriever(
        vectorstore=self.vector,
        docstore=self.store,
        child_splitter=self.splitter,
        parent_splitter=self.parent_splitter,)
        self.retriever.add_documents(self.docs)
    
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
        documents = self.splitter.split_documents(self.docs[1:])
        self.vector = FAISS.from_documents(documents, self.embeddings)
        self.vector.save_local(self.db_path)
  
    
    def load_db(self):
        self.vector=FAISS.load_local(self.db_path,self.embeddings,allow_dangerous_deserialization=True)
 
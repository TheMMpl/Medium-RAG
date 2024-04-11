import argparse
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from vector_database import VectorDatabase
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="mode - local or external model", type=str)
    parser.add_argument("--query", help="query to be answered", type=str)
    parser.add_argument("--display_db_results", help="view results passed to the llm", type=str)
    args = parser.parse_args()

    
    db=VectorDatabase(args.mode)
    if args.mode=="local":
        llm=Ollama(model="llama2")
    else:
        llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_template("""You are an assistant hepling with explaining machine learning techniques. Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")


    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": args.query})
    print(response["answer"])

    if args.display_db_results=="True":
        print(db.similarity_search_with_score(args.query))



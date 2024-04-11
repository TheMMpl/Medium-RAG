# Medium-RAG
## Installation
clone the repository and install the requirements by running
```
pip install -r requirements. txt
```
For use in local mode you must install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```
and then download a quantized llama2 model
```
ollama pull llama2
```
and an embedding model 
```
ollama pull mxbai-embed-large
```
## Usage
This RAG demo can be run in two modes - local or external.
For the external mode you must provide an OpenAI api key by executing
```
export OPENAI_API_KEY="your api key"
```
Then you can ask a question by using
```
python3 medium_rag.py --mode external --query "your question here"
```
Similarily, in local mode you can ask a question by using 
```
python3 medium_rag.py --mode local --query "your question here" 
```
If used for the first time the script will create a faiss database with the embedded chunks.

There are optional parameters availible:

`--display_db_results` when `True` displays top results from database search.

`--chunking_strategy` in `Parent` mode Parent document retrieval is used allowing the extraction of larger chunks containing shoreter ones stored directly in the vector database. This impacts performance as the large chunks must be initalized each time the command is ran.

`--question_type` adjusts the base size of chunks - smaller chunks are used for `specific` questions, where the answer might be contained directly in a similar sentence, and larger chunks are used for `general` questions.

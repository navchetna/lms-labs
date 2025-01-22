# this is the next stage in the rag pipeline where there are different chunking techniques being used

# there are two parts to it, the first is creating the embeddings, the second is to ask a question against the stored embeddings. 

# In the create embeddings flow. 
### the first step is to read an input pdf pass it through the reader that reads page at a time and converts it into string and returns a list of strings
### the second step is to create chunks out of the input list of strings passed to it. chunks decided by their size, done by a SimpleChunker. output is a list of chunks, and list of Documents that is chunks plus the document meta-data
### the thrid step is to create embeddings
### the fourth step is to store them in a vector database

# In the query flow
### The for the input query first embeddings are created
### The next step is to retrieve the chunks that have a similarity to the query-embeddings
### These are then passed into the LLMs to format a response 

### Running the Rag-pipeline 
Install the dependencies in a virtual environment and run the files read-file-adapt-create-embeddings-and-write.py followed by ask_chatbot.py
```bash
$ source activate rag_env
$ pip install -r requirements.txt
$ python read-file-adapt-create-embeddings-and-write.py
$ python ask_chatbot.py
```
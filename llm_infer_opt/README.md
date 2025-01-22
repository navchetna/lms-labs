### Inferencing with the GGUF models (llm_gguf.py)  
Install the dependencies in a virtual environment and execute the python script llm_gguf.py
```bash 
$ source activate llm_inf
$ pip install -r requirements.txt
$ python llm_gguf.py --model-name mistralai/Mistral-7B-Instruct-v0.2  -p "What do you mean by generative artificial intelligence?"  
```

### Inferencing with GGUF models using Intels Extension for transformers(llm_gguf_NS.py)
Install the dependencies in a virtual environment and execute the python script llm_gguf_NS.py
```bash 
$ source activate llm_inf
$ pip install -r requirements.txt
$ python llm_gguf_NS.py --model-name mistralai/Mistral-7B-Instruct-v0.2 -p "What do you mean by generative artificial intelligence?" 
```
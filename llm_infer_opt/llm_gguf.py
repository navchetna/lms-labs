import os  
import time 
import argparse
import psutil
from llama_cpp import Llama


MODELS = {'meta-llama/Llama-2-7b-chat-hf': '/models/llama-2-7b-chat.Q4_0.gguf', 'mistralai/Mistral-7B-Instruct-v0.2': '/models/mistral-7b-instruct-v0.2.Q4_0.gguf'}


class LLM_CPP():
    def __init__(self, model_name): 
        self.model = Llama(
      model_path= MODELS[model_name],
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
)
        
    def generate(self, prompt: str, max_new_tokens: int = 64): 
        start_time = time.perf_counter()
        output = self.model(prompt, max_tokens = max_new_tokens) 
        total_time = time.perf_counter() - start_time 
        
        # Only decode and return the model generated tokens 
        generated_text = output['choices'][0]['text']
        
        self._print_metrics(output['usage']['prompt_tokens'], output['usage']['completion_tokens'], total_time, generated_text) 
        return generated_text 
    
    def generate_batched(self, prompt: str, max_new_tokens: int = 64):
        """Under development; Support - https://github.com/abetlen/llama-cpp-python/blob/7d4a5ec59f0cf11d36e6f883e81f9f3859e13280/examples/notebooks/Batching.ipynb""" 
        start_time = time.perf_counter()
        output = self.model([prompt]*2, max_tokens = max_new_tokens) 
        total_time = time.perf_counter() - start_time 
        
        # Only decode and return the model generated tokens 
        generated_text = output['choices'][0]['text']
        
        self._print_metrics(output['usage']['prompt_tokens'], output['usage']['completion_tokens'], total_time, generated_text) 
        return generated_text
    
    def _print_metrics(self, inputs, outputs, total_time, generated_text): 
        print("\n\n" + "-"*20)
        print(f"Input Tokens: {inputs}")
        print(f"Output Tokens: {outputs}")
        print(f"Total Time: {total_time}")
        print(f"Tokens per second : {outputs/total_time}")
        print(f"Generated Text : {generated_text}")
        print("-"*20)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LLM Inferencing using ITREX")
    
    parser.add_argument('--model-name', help = "Enter the model name")
    parser.add_argument('-p', '--prompt', help = "Prompt to the model", required = True, default = "What does generative artificial intelligence mean?")
    parser.add_argument("--ctx-size", help = "Maximum prompt length", default = 64)
    
    args = parser.parse_args()
    
    
    current_process = psutil.Process(os.getpid())
    
    
    before_memory_info = current_process.memory_info()
    llm = LLM_CPP(args.model_name)
    output = llm.generate(args.prompt)
    after_memory_info = current_process.memory_info()
    
    # Calculate the difference in memory usage
    memory_used = after_memory_info.rss - before_memory_info.rss
    
    print(f"Memory used by : {memory_used/(1024 ** 3)} GB")
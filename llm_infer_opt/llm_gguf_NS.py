import os  
import time 
import argparse
import psutil 
from transformers import AutoTokenizer, TextStreamer 
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, AwqConfig, RtnConfig, GPTQConfig, AutoRoundConfig


MODELS = {'meta-llama/Llama-2-7b-chat-hf': {"model_file":"llama-2-7b-chat.Q4_0.gguf", "model_name": "TheBloke/Llama-2-7B-Chat-GGUF"}, 
          'mistralai/Mistral-7B-Instruct-v0.2': {'model_file': "mistral-7b-instruct-v0.2.Q4_0.gguf", "model_name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"}}



class LLM_GGUF_NS():
    def __init__(self, model_name = "meta-llama/Llama-2-7b-chat-hf", ): 
        self.model = AutoModelForCausalLM.from_pretrained(MODELS[model_name]['model_name'], model_file = MODELS[model_name]['model_file'], use_cache = True, use_neural_speed = False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True) 
        self.streamer = TextStreamer(self.tokenizer) 
        
    def generate(self, prompt: str, max_new_tokens: int = 64): 
        inputs = self.tokenizer(prompt, return_tensors = "pt").input_ids
        start_time = time.perf_counter()
        output = self.model.generate(inputs, streamer = self.streamer, max_new_tokens = max_new_tokens) 
        total_time = time.perf_counter() - start_time 
        
        # Only decode and return the model generated tokens 
        generated_text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        
        self._print_metrics(inputs, output, total_time, generated_text) 
        return generated_text 
    
    def generate_batched(self, prompt: str, max_new_tokens: int = 64): 
        """Currently doesn't support batching!"""
        
        inputs = self.tokenizer(prompt, return_tensors = "pt").input_ids
        start_time = time.perf_counter()
        output = self.model.generate(inputs, streamer = self.streamer, max_new_tokens = max_new_tokens) 
        total_time = time.perf_counter() - start_time 
        
        # Only decode and return the model generated tokens 
        generated_text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        
        self._print_metrics(inputs, output, total_time, generated_text) 
        return generated_text 
    
    def _print_metrics(self, inputs, outputs, total_time, generated_text): 
        generated_tokens = len(outputs[0]) - len(inputs[0]) 
        print("\n\n" + "-"*20)
        print(f"Input Tokens: {len(inputs[0])}")
        print(f"Output Tokens: {generated_tokens}")
        print(f"Total Time: {total_time}")
        print(f"Tokens per second : {generated_tokens/total_time}")
        print(f"Generated Text : {generated_text}")
        print("-"*20)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LLM Inferencing using ITREX")
    
    parser.add_argument("--model-name", help = "Name of the model", choices = MODELS, required = True)
    parser.add_argument('-p', '--prompt', help = "Prompt to the model", required = True, default = "What does generative artificial intelligence mean?")
    parser.add_argument("--ctx-size", help = "Maximum prompt length", default = 64)
    
    args = parser.parse_args()
    
    
    current_process = psutil.Process(os.getpid())
    
    
    before_memory_info = current_process.memory_info()
    llm = LLM_GGUF_NS(args.model_name)
    output = llm.generate(args.prompt)
    after_memory_info = current_process.memory_info()
    
    # Calculate the difference in memory usage
    memory_used = after_memory_info.rss - before_memory_info.rss
    
    print(f"Memory used by : {memory_used/(1024 ** 3)} GB")
import torch
from transformers import AutoModel, AutoTokenizer

class DiffuCoderInference:
    """
    A class for running inference with the DiffuCoder-7B-Instruct model.
    """
    
    def __init__(self, model_path="apple/DiffuCoder-7B-Instruct"):
        """
        Initialize the DiffuCoder model and tokenizer.
        
        Args:
            model_path (str): Path to the model on HuggingFace
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = self.model.to(self.device).eval()
        print(f"Model loaded successfully on {self.device}")
    
    def generate_response(self, query, max_new_tokens=256, temperature=0.4, top_p=0.95, steps=None):
        """
        Generate a response for the given query using the DiffuCoder model.
        
        Args:
            query (str): The input query/prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for sampling
            top_p (float): Top-p value for nucleus sampling
            steps (int): Number of diffusion steps. If None, calculated as max_new_tokens // TOKEN_PER_STEP
            
        Returns:
            str: Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create the prompt template (following qwen format)
        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{query.strip()}
<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # Set default steps if not provided
        TOKEN_PER_STEP = 1
        if steps is None:
            steps = max_new_tokens // TOKEN_PER_STEP
        
        # Generate response
        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg="entropy",
            alg_temp=0.,
        )
        
        # Decode the generation
        generations = [
            self.tokenizer.decode(g[len(p):].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]
        
        # Clean up the response (remove padding and end tokens)
        response = generations[0].split('<|dlm_pad|>')[0]
        response = response.split('<|im_end|>')[0]
        return response.strip()


def create_inference_function():
    """
    Create a simple function interface for inference.
    This loads the model once and returns a function that can be called multiple times.
    """
    diffucoder = DiffuCoderInference()
    
    def inference_function(prompt, **kwargs):
        """
        Simple function to generate response for a given prompt.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments passed to generate_response
            
        Returns:
            str: Generated response
        """
        return diffucoder.generate_response(prompt, **kwargs)
    
    return inference_function


# Example usage
if __name__ == "__main__":
    print("Testing DiffuCoder inference...")
    
    inference_func = create_inference_function()
    
    test_query = "What is 2 + 2 - 5?"
    
    print(f"Query: {test_query}")
    
    response = inference_func(test_query)
    print(f"Response: {response}") 
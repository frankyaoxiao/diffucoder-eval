"""
Custom Inspect AI Model API for DiffuCoder-7B-Instruct
"""
import torch
from typing import Any, List
from inspect_ai.model import ModelAPI, GenerateConfig, ChatMessage, ModelOutput, ChatMessageUser, ChatMessageAssistant
from inspect_ai.tool import ToolInfo, ToolChoice
from transformers import AutoModel, AutoTokenizer


class DiffuCoderModelAPI(ModelAPI):
    """Custom ModelAPI implementation for DiffuCoder-7B-Instruct"""
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        
        # Initialize the DiffuCoder model
        self.model_path = "apple/DiffuCoder-7B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Store any custom model arguments
        self.model_args = model_args
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the DiffuCoder model and tokenizer"""
        print(f"Loading DiffuCoder model...")
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
        print(f"DiffuCoder model loaded successfully on {self.device}")
    
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the DiffuCoder model"""
        
        # Convert Inspect AI messages to the format expected by DiffuCoder
        prompt = self._convert_messages_to_prompt(input)
        
        # Get generation parameters from config
        max_new_tokens = getattr(config, 'max_tokens', 256) or 256
        temperature = getattr(config, 'temperature', 0.4) or 0.4
        top_p = getattr(config, 'top_p', 0.95) or 0.95
        
        # Generate response using DiffuCoder
        response_text = await self._generate_response(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Create ModelOutput
        return ModelOutput.from_content(
            model=self.model_name,
            content=response_text
        )
    
    def _convert_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert Inspect AI ChatMessage list to DiffuCoder prompt format"""
        
        # Start with system message
        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
"""
        
        # Process messages
        for message in messages:
            if isinstance(message, ChatMessageUser):
                content = message.content
                if isinstance(content, list):
                    # Handle multimodal content - extract text
                    text_parts = [item.text for item in content if hasattr(item, 'text')]
                    content = " ".join(text_parts)
                
                prompt += f"""<|im_start|>user
{content.strip()}
<|im_end|>
"""
            elif isinstance(message, ChatMessageAssistant):
                content = message.content or ""
                prompt += f"""<|im_start|>assistant
{content.strip()}
<|im_end|>
"""
        
        # Add the assistant start for the new response
        prompt += """<|im_start|>assistant
"""
        
        return prompt
    
    def _generate_response_sync(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.4,
        top_p: float = 0.95
    ) -> str:
        """Generate response using the loaded DiffuCoder model (synchronous)"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # Set diffusion steps
        TOKEN_PER_STEP = 1
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
    
    async def _generate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.4,
        top_p: float = 0.95
    ) -> str:
        """Generate response using the loaded DiffuCoder model (async wrapper)"""
        # Run the synchronous generation in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._generate_response_sync, 
            prompt, 
            max_new_tokens, 
            temperature, 
            top_p
        )


# Model provider registration - simplified approach
from inspect_ai.model import modelapi

class DiffuCoderProvider:
    """DiffuCoder model provider that handles any model ID"""
    
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ):
        # Create the actual model API instance
        # Inspect AI passes the full model name like "diffucoder/7b-instruct"
        self.model_api = DiffuCoderModelAPI(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars,
            config=config,
            **model_args
        )
    
    def __getattr__(self, name):
        # Delegate all method calls to the underlying model API
        return getattr(self.model_api, name)

@modelapi(name="diffucoder")
def diffucoder_provider():
    """Create DiffuCoder provider"""
    return DiffuCoderProvider

print("DiffuCoder provider registered successfully!") 
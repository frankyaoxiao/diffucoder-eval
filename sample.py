# sample eval script from apple
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "apple/DiffuCoder-7B-Instruct"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

query = "Write a function to find the shared elements from the given two lists."
prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{query.strip()}
<|im_end|>
<|im_start|>assistant
""" ## following the template of qwen; you can also use apply_chat_template function

TOKEN_PER_STEP = 1 # diffusion timesteps * TOKEN_PER_STEP = total new tokens

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    output_history=True,
    return_dict_in_generate=True,
    steps=256//TOKEN_PER_STEP,
    temperature=0.4,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split('<|dlm_pad|>')[0])
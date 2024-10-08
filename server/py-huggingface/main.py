import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if torch.backends.mps.is_available():
    active_device = torch.device("mps")
elif torch.cuda.is_available():
    active_device = torch.device("cuda", 0)
else:
    active_device = torch.device("cpu")

accelerator = Accelerator()
print("Accelerator device: ", accelerator.device)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map=active_device,
    torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to(active_device)

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))

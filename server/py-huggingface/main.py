import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


# init: huggingface
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# init: fastapi
app = FastAPI(title="py-huggingface")


class RequestItem(BaseModel):
    role: str
    content: str


# routes
@app.post("/")
async def main(request: RequestItem):
    input_ids = tokenizer(request.content, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=2048)

    return {"content": tokenizer.decode(outputs[0])}

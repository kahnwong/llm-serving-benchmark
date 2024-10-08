import torch
import transformers
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI(title="py-huggingface")

model = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


class RequestItem(BaseModel):
    role: str
    content: str


@app.post("/")
async def main(request: RequestItem):
    sequences = pipeline(
        request.content,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
    )

    response = ""
    for seq in sequences:
        response += seq["generated_text"]

    return {"content": response}

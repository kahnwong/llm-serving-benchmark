import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

app = FastAPI(title="py-huggingface")


class RequestItem(BaseModel):
    role: str
    content: str


@app.post("/")
async def main(request: RequestItem):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {"role": "user", "content": request.content},
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    return {"content": (outputs[0]["generated_text"])}

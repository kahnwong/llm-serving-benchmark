cpp-llamafile-start:
	./server/cpp-llamafile/TinyLlama-1.1B-Chat-v1.0.Q4_0.llamafile
go-ollama-start:
	ollama run tinyllama:1.1b-chat
py-huggingface-start:
	cd server/py-huggingface && uv run uvicorn main:app --port 8080

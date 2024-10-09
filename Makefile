cpp-llamafile-start:
	./server/cpp-llamafile/gemma-2-2b-it.Q4_0.llamafile
# go-ollama-start:  # ollama is a daemon
# 	ollama run gemma2:2b-instruct-q4_0t
py-huggingface-start:
	cd server/py-huggingface && uv run uvicorn main:app --port 8080

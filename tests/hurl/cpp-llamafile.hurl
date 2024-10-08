POST http://localhost:8080/v1/chat/completions
Content-Type: application/json
Authorization: Bearer no-key
{
  "model": "LLaMA_CPP",
  "messages": [
      {
          "role": "user",
          "content": "How many helicopters can a human eat in one sitting?"
      }
    ]
}
HTTP 200

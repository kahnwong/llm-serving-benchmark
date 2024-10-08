POST http://localhost:8080/v1/chat/completions
Content-Type: application/json
Authorization: Bearer no-key
{
  "model": "LLaMA_CPP",
  "messages": [
      {
          "role": "user",
          "content": "Write a limerick about python exceptions"
      }
    ]
}
HTTP 200

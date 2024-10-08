POST http://localhost:11434/api/chat
Content-Type: application/json
{
  "model": "llama3:8b-instruct-q4_0",
  "stream": false,
  "messages": [
      {
          "role": "user",
          "content": "Write a limerick about python exceptions"
      }
    ]
}
HTTP 200

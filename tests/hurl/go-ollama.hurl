POST http://localhost:11434/api/chat
Content-Type: application/json
{
  "model": "gemma2:2b-instruct-q4_0",
  "stream": false,
  "messages": [
      {
          "role": "user",
          "content": "How many helicopters can a human eat in one sitting?"
      }
    ]
}
HTTP 200

POST http://localhost:11434/api/chat
Content-Type: application/json
{
  "model": "tinyllama:1.1b-chat",
  "stream": false,
  "messages": [
      {
          "role": "user",
          "content": "How many helicopters can a human eat in one sitting?"
      }
    ]
}
HTTP 200

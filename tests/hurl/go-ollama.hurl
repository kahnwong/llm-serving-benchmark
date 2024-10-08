POST http://localhost:11434/api/chat
Content-Type: application/json
{
  "model": "tinyllama:1.1b-chat",
  "stream": false,
  "messages": [
      {
          "role": "user",
          "content": "Write a limerick about python exceptions"
      }
    ]
}
HTTP 200

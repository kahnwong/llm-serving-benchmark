POST http://localhost:1234/v1/chat/completions
Content-Type: application/json
{
  "model": "mistral.rs",
  "messages": [
      {
          "role": "user",
          "content": "How many helicopters can a human eat in one sitting?"
      }
    ]
}
HTTP 200

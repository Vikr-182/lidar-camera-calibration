import os
import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {
      "role": "user",
      "content": "hi hlelo\n"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I assist you today?\n"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)
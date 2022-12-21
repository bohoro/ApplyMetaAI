# Set up the OpenAI ChatGPT API client
import os

import openai

openai.api_key = os.environ["OAI_TOKEN"]

# Get GPT Completion
def compare_responses(prompt, length=1024):
    chatbot_response = (
        openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        .choices[0]
        .text
    )
    return chatbot_response


print(compare_responses("Wikipedia was launched by"))

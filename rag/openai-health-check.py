import openai
import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

openai_client=openai.Client(api_key=api_key)
list=openai_client.models.list();
print('connected to open ai')

completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )

answer = completion.choices[0].message.content
print(f"✅ OpenAI Response: {answer}")
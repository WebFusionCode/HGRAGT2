import asyncio
from litellm import acompletion
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    try:
        response = await acompletion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print("Success:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", str(e))

asyncio.run(test())

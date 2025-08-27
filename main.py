import os
from dotenv import load_dotenv
from groq import Groq
from rag import main as rag_main   # import your RAG entrypoint


# Load variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Who are you",
        },
        {
            "role": "system",
            "content": "You are a e-commerce competitor analyst",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print("\n🤖 Groq test reply:")
print(chat_completion.choices[0].message.content)

# ─────────── Run RAG ───────────
print("\n🚀 Running RAG pipeline:\n")
rag_main()
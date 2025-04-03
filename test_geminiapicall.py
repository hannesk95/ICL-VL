import google.generativeai as genai
from dotenv import load_dotenv
import os
import re

# Load the API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Please set it in .env or your environment.")

# Configure the SDK
genai.configure(api_key=GEMINI_API_KEY)

# Call Gemini
def gemini_api_call(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content(prompt)
    return response.text

# Test
print("Gemini Model Prediction:", gemini_api_call("What is the capital of France?"))
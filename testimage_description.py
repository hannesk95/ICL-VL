import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
from PIL import Image

# Load the API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Please set it in .env or your environment.")

# Configure the SDK
genai.configure(api_key=GEMINI_API_KEY)

# Load image and describe it
def describe_image(image_path, prompt="Describe this image"):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Use a model that supports multimodal input
    with Image.open(image_path) as img:
        response = model.generate_content([prompt, img])
    return response.text

# Test it with an image
image_path = "/u/home/obt/ICL-VL/data/negative/1 no.jpeg"  # Replace with your actual image path
print("Image Description:", describe_image(image_path))
import google.generativeai as genai
from PIL import Image 
import os

API_KEY = "AIzaSyAD0bbYAyELrfqdmSb64d3dVcGT5SqRbkA" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

image_path = "/u/home/obt/ICL-VL/ADI-AAAMHQMK.png"

# Check if the image exists
if not os.path.exists(image_path):
    print(f"❌ ERROR: Image file '{image_path}' not found.")
    exit(1)

image = Image.open(image_path)

# Send image and prompt to the Gemini Vision API
print("🟡 Processing image...")

try:
    response = model.generate_content([image, "Describe this image in detail."])
    print("✅ Response Received:")
    print(response.text)
except Exception as e:
    print(f"❌ API Error: {e}")
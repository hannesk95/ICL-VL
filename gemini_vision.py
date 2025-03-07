from google import genai
from google.genai import types
import PIL
import io

if __name__ == "__main__":

    api_key = "paste here"

    # Read image from file
    path = "curch_organ.jpg"
    path = "dog.png"
    image = PIL.Image.open(path)

    # Convert PIL Image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format or "PNG")  # Use image.format or default to PNG if None
    image_bytes = image_bytes.getvalue()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        # model="gemini-2.0-flash-exp",
        contents=["What is this image?", 
                types.Part.from_bytes(data=image_bytes, mime_type=f"image/{image.format.lower() if image.format else 'png'}")]  # Use correct MIME type
    )

    print(response.text)

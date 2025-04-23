import google.generativeai as genai
import pprint

genai.configure(api_key='AIzaSyAD0bbYAyELrfqdmSb64d3dVcGT5SqRbkA')

"""for model in genai.list_models():
    pprint.pprint(model)"""
    
# The models you want to check
models_to_check = [
    "models/gemini-2.5-pro-preview-03-25",
    "models/gemini-2.5-pro-exp-03-25"
]

# Try getting metadata for each model
for model_name in models_to_check:
    try:
        model = genai.get_model(model_name)
        print(f"✅ You have access to: {model_name}")
    except Exception as e:
        print(f"❌ No access to: {model_name} — {e}")
        
# model I use gemini-2.0-flash
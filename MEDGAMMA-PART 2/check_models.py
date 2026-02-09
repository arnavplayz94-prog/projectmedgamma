import google.generativeai as genai
import os
from bby import GEMINI_API_KEY

if not GEMINI_API_KEY:
    print("No API Key found")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        print("Listing available models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error: {e}")

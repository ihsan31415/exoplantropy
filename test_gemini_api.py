"""Test script to verify Gemini API key"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from config/gemini.env
load_dotenv(dotenv_path="config/gemini.env", override=True)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY tidak ditemukan!")
    print("Pastikan file config/gemini.env berisi:")
    print("GEMINI_API_KEY=<your_api_key>")
else:
    print(f"✓ API Key ditemukan: {api_key[:20]}...")
    
    try:
        genai.configure(api_key=api_key)
        
        # List available models
        print("\n📋 Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
        
        # Test with correct model name (use the full path with 'models/' prefix)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content("Say hello in one sentence!")
        print(f"\n✅ API Key VALID!")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ API Key INVALID atau ada error!")
        print(f"Error: {e}")

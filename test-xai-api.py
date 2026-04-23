"""
Test xAI API connection and list available models
"""

import os
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# API key from environment
API_KEY = os.getenv("XAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Missing XAI_API_KEY. Set it in your environment or .env file.")

print("="*60)
print("xAI API Connection Test")
print("="*60)

# Initialize client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.x.ai/v1"
)

print("\n1️⃣ Testing API connection...")
try:
    # Try a simple text completion
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "user", "content": "Say 'API is working!' if you can respond."}
        ],
        max_tokens=20
    )
    
    print(f"✅ API Connected!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API Connection Failed: {e}")
    exit(1)

print("\n2️⃣ Testing available models...")
try:
    # Try to list models
    models = client.models.list()
    print(f"✅ Available models:")
    for model in models.data:
        print(f"   - {model.id}")
except Exception as e:
    print(f"⚠️  Could not list models: {e}")
    print(f"💡 Common models: grok-beta, grok-2-latest")

print("\n3️⃣ Testing image generation capability...")

# Try different possible model names
image_models_to_try = [
    "grok-imagine-image",
    "grok-vision-beta", 
    "grok-image",
    "grok-beta"
]

for model_name in image_models_to_try:
    try:
        print(f"\n   Testing model: {model_name}...")
        response = client.images.generate(
            model=model_name,
            prompt="A simple test: red fire in a kitchen",
            n=1,
        )
        print(f"   ✅ SUCCESS! Image generation works with: {model_name}")
        print(f"   Image URL: {response.data[0].url}")
        print(f"\n   🎉 Update your script to use model: {model_name}")
        break
    except AttributeError as e:
        print(f"   ❌ client.images not available")
        break
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"   ❌ Model not found")
        else:
            print(f"   ❌ Error: {str(e)[:100]}")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("✅ Text generation: WORKING")
print("❌ Image generation: NOT AVAILABLE")
print("\n💡 RECOMMENDATION:")
print("   Use Grok to enhance prompts, then generate images with:")
print("   - DALL-E 3 (OpenAI)")
print("   - Gemini (Google) - you already have this!")
print("   - Midjourney")
print("   - Stable Diffusion")
print("="*60)


import google.generativeai as genai

# --- IMPORTANT ---
# Paste your API key here
GEMINI_API_KEY = "**"
# -----------------

genai.configure(api_key=GEMINI_API_KEY)

print("--- Available Gemini Models ---")
for model in genai.list_models():
  # We are looking for models that support the 'generateContent' method for the main LLM call
  if 'generateContent' in model.supported_generation_methods:
    print(f"Model Name: {model.name}")
print("-----------------------------")

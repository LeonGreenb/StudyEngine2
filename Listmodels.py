import google.generativeai as genai

    # Configure your API key
genai.configure(api_key="AIzaSyBjvJDhD63rrRXTBZoQtX2Ne4gsloHJvAM")

    # List available models
for m in genai.list_models():
        print(f"Model: {m.name}, Supported Methods: {m.supported_generation_methods}")
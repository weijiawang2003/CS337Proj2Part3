from dotenv import load_dotenv
import os
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Example request to the Gemini model
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words."
)

# Print the model's response text
print(response.text)
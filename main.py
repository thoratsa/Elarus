import requests
import time
import json
import os
from flask import Flask, request, jsonify

# --- Configuration ---
# NOTE: For local testing, you MUST set the GROQ_API_KEY environment variable.
# Example: export GROQ_API_KEY='YOUR_API_KEY_HERE'
GROQ_MODEL = "openai/gpt-oss-120b"
API_KEY = os.environ.get("GROQ_API_KEY", "") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Exponential Backoff Configuration
MAX_RETRIES = 5
BASE_DELAY = 1  # seconds

app = Flask(__name__)

# Check for API key if running locally
if not API_KEY and '127.0.0.1' in os.environ.get('FLASK_RUN_HOST', '127.0.0.1'):
    print("="*60)
    print("WARNING: GROQ_API_KEY environment variable is not set.")
    print("The API call will likely fail with a 401 or 403 error.")
    print("To test locally with curl, please set the environment variable:")
    print(f"export GROQ_API_KEY='YOUR_API_KEY_HERE'")
    print("="*60)


def call_groq_api_with_backoff(system_instruction, user_prompt):
    """
    Calls the Groq API (using OpenAI format) with exponential backoff retry logic.
    """
    current_delay = BASE_DELAY
    
    # Groq uses the standard Chat Completions format with a list of messages
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2, # Low temperature for reliable translation
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            data = response.json()
            
            # Extract the raw text from the Groq/OpenAI response structure
            if data and data.get('choices'):
                text_content = data['choices'][0]['message']['content']
                
                # Strip any leading/trailing whitespace
                return text_content.strip()
            
            # If structure is valid but no text is returned, treat as an unexpected error
            raise requests.exceptions.HTTPError(f"API returned no text content: {data}")

        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            # Handle transient errors (like 429 Rate Limit or 5xx Server Errors)
            if attempt < MAX_RETRIES - 1:
                print(f"API call failed (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
            else:
                # Last attempt failed
                print(f"API call failed after {MAX_RETRIES} attempts. Final error: {e}")
                raise

    # This line should ideally not be reached, but is a fallback
    raise Exception("Translation failed after maximum retries.")


@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Flask API endpoint for translation.
    Expects JSON payload: {"text": "...", "target_lang": "...", "source_lang": "..." (optional)}
    """
    try:
        data = request.get_json()
        
        # Validate required inputs
        text_to_translate = data.get('text')
        target_lang = data.get('target_lang')
        source_lang = data.get('source_lang', 'the source language (auto-detect if unknown)')

        if not text_to_translate or not target_lang:
            return jsonify({"error": "Missing 'text' or 'target_lang' in request payload."}), 400

        # Prompt Engineering for raw output
        system_instruction = (
            f"You are a professional, high-quality language translator. Your task is to translate the provided text "
            f"from {source_lang} to {target_lang}. "
            f"IMPORTANT: You MUST ONLY return the translated text and NOTHING else. "
            f"Do not include any introductory phrases, explanations, markdown formatting (like quotes or bolding), or punctuation beyond what is in the translation."
        )

        # The user_prompt just holds the raw text for the user message role
        user_prompt = text_to_translate 

        translated_text = call_groq_api_with_backoff(system_instruction, user_prompt)
        
        return jsonify({
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        })

    except Exception as e:
        # Catch any unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred during translation.", "details": str(e)}), 500

if __name__ == '__main__':
    # Flask is run in debug mode for development purposes.
    # Setting host='0.0.0.0' makes it accessible externally in containers/VMs
    print(f"Starting API server (using Groq API with {GROQ_MODEL}).")
    print(f"Access the /translate endpoint via POST request at http://127.0.0.1:5000/translate")
    print("Example Payload: {'text': 'Hello world', 'target_lang': 'French'}")
    app.run(debug=True, host='0.0.0.0', port=5000)

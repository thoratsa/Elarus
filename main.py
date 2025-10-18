import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"

@app.route('/translate', methods=['POST'])
def translate():
    if not API_KEY:
        return jsonify({"error": "API key not configured."}), 500

    try:
        data = request.get_json()
        text_to_translate = data.get('text')
        target_lang = data.get('target_lang')
        
        if not text_to_translate or not target_lang:
            return jsonify({"error": "Missing 'text' or 'target_lang' in request body."}), 400

    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    system_prompt = f"You are a professional translator. Translate the user's text into {target_lang}. Only return the translated text. Do not include any explanations, greetings, or punctuation outside of the translation itself."
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user": "content": text_to_translate}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        groq_response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        groq_response.raise_for_status()
        groq_data = groq_response.json()

        translated_text = groq_data['choices'][0]['message']['content'].strip()

        return jsonify({
            "source_language": "auto-detected",
            "target_language": target_lang,
            "translated_text": translated_text
        })

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        return jsonify({"error": f"Groq API Error ({status_code}): {error_details}"}), status_code

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during translation: {str(e)}"}), 500

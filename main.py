import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

CACHE_FILE = 'translation_cache.json'


app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "GPT-OSS-120B"


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}

def save_cache(cache_data):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"ERROR: Could not save cache file. Persistence failed. {e}")


@app.route('/translate', methods=['POST'])
def translate():
    if not API_KEY:
        return jsonify({"error": "API key not configured."}), 500

    try:
        data = request.get_json()
        text_to_translate = data.get('text', '').strip()
        target_lang = data.get('target_lang', '').strip()
        
        if not text_to_translate or not target_lang:
            return jsonify({"error": "Missing 'text' or 'target_lang' in request body."}), 400

    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    
    cache = load_cache()
    cache_key = f"{text_to_translate}|{target_lang}"
    
    if cache_key in cache:
        cached_result = cache[cache_key]
        print("Cache HIT!")
        return jsonify({
            "source_language": cached_result['source_language'],
            "target_language": target_lang,
            "translated_text": cached_result['translated_text'],
            "status": "cached"
        })

    print("Cache MISS! Calling Groq...")
    system_prompt = f"You are a professional translator. Translate the user's text into {target_lang}. Only return the translated text. Do not include any explanations, greetings, or punctuation outside of the translation itself."
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_to_translate}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        groq_response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        groq_response.raise_for_status()
        groq_data = groq_response.json()

        translated_text = groq_data['choices'][0]['message']['content'].strip()

        response_data = {
            "source_language": "auto-detected",
            "target_language": target_lang,
            "translated_text": translated_text,
            "status": "generated"
        }
        
        cache[cache_key] = {
            "source_language": "auto-detected",
            "translated_text": translated_text,
            "timestamp": time.time(),
            "model": MODEL_NAME
        }
        save_cache(cache)
        
        return jsonify(response_data)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        return jsonify({"error": f"Groq API Error ({status_code}): {error_details}"}), status_code

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during translation: {str(e)}"}), 500

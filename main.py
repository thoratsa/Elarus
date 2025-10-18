import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import time

REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("Redis connection successful.")
    except Exception as e:
        print(f"Redis connection failed: {e}. Caching disabled.")
        r = None

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 


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

    
    cache_key = f"translation:{text_to_translate}|{target_lang}"
    
    if r:
        try:
            cached_result_json = r.get(cache_key)
            if cached_result_json:
                cached_result = json.loads(cached_result_json)
                print("Cache HIT!")
                return jsonify({
                    "source_language": cached_result['source_language'],
                    "target_language": target_lang,
                    "translated_text": cached_result['translated_text'],
                    "status": "cached"
                })
        except Exception as e:
            print(f"Error reading from Redis: {e}")

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
        
        if r:
            try:
                cache_data = {
                    "source_language": "auto-detected",
                    "translated_text": translated_text,
                    "timestamp": time.time(),
                    "model": MODEL_NAME
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
                print("Successfully saved result to Redis.")
            except Exception as e:
                print(f"Error writing to Redis: {e}")
        
        return jsonify(response_data)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        return jsonify({"error": f"Groq API Error ({status_code}): {error_details}"}), status_code

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during translation: {str(e)}"}), 500

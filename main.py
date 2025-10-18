import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception as e:
        r = None

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 
RATE_LIMIT_SECONDS = 1 


def check_rate_limit(client_id):
    if not r:
        return True
    
    current_time = time.time()
    key = f"rate_limit:{client_id}"
    
    try:
        last_request_time = r.get(key)
        if last_request_time:
            time_since_last = current_time - float(last_request_time)
            if time_since_last < RATE_LIMIT_SECONDS:
                return False
        
        r.set(key, current_time, ex=RATE_LIMIT_SECONDS * 2)
        return True
    except Exception as e:
        return True

def get_source_language(text):
    try:
        return detect(text).upper()
    except LangDetectException:
        return "Unknown"

def _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False):
    if not check_rate_limit(client_ip):
        return jsonify({"error": f"Rate limit exceeded. Try again in {RATE_LIMIT_SECONDS} second(s)."}), 429

    source_language_code = get_source_language(text_to_translate)
    cache_key = f"translation:{text_to_translate}|{target_lang}"

    if not force_refresh and r:
        try:
            cached_result_json = r.get(cache_key)
            if cached_result_json:
                cached_result = json.loads(cached_result_json)
                return jsonify({
                    "source_language": cached_result['source_language'],
                    "target_language": target_lang,
                    "translated_text": cached_result['translated_text'],
                    "status": "cached"
                }), 200
        except Exception as e:
            pass
    
    max_tokens_calculated = min(2000, len(text_to_translate) * 2 + 50) 
    
    status_type = "reviewed" if force_refresh else "generated"

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
        "temperature": 0.1,
        "max_tokens": max_tokens_calculated
    }
    
    try:
        groq_response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        groq_response.raise_for_status()
        groq_data = groq_response.json()

        translated_text = groq_data['choices'][0]['message']['content'].strip()
        
        response_data = {
            "source_language": source_language_code,
            "target_language": target_lang,
            "translated_text": translated_text,
            "status": status_type
        }
        
        if r:
            try:
                cache_data = {
                    "source_language": source_language_code,
                    "translated_text": translated_text,
                    "timestamp": time.time(),
                    "model": MODEL_NAME
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
            except Exception as e:
                pass
        
        return jsonify(response_data), 200

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        return jsonify({"error": f"Groq API Error ({status_code}): {error_details}"}), status_code

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during translation: {str(e)}"}), 500

@app.route('/translate', methods=['POST'])
def translate():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()
    
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
    
    return _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False)

@app.route('/review', methods=['POST'])
def review():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()
    
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
    
    return _process_translation(text_to_translate, target_lang, client_ip, force_refresh=True)

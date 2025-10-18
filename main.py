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

# --- Configuration ---
REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')
GROQ_MODEL = "openai/gpt-oss-120b"
API_KEY = os.environ.get("GROQ_API_KEY", "") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 
RATE_LIMIT_SECONDS = 1
MAX_RETRIES = 5
BASE_DELAY = 1

# --- Redis Initialization ---
r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception:
        r = None

app = Flask(__name__)
CORS(app)

# --- Local Warning ---
if not API_KEY and '127.0.0.1' in os.environ.get('FLASK_RUN_HOST', '127.0.0.1'):
    print("="*60)
    print("WARNING: GROQ_API_KEY environment variable is not set.")
    print("The API call will likely fail with a 401 or 403 error.")
    print("To test locally with curl, please set the environment variable:")
    print(f"export GROQ_API_KEY='YOUR_API_KEY_HERE'")
    print("="*60)

# --- Utility Functions ---

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
    except Exception:
        return True

def get_source_language(text):
    try:
        return detect(text).upper()
    except LangDetectException:
        return "Unknown"

def _get_client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()

# --- Core LLM Function with Backoff ---

def call_groq_api_with_backoff(system_instruction, user_prompt):
    current_delay = BASE_DELAY
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            data = response.json()
            
            if data and data.get('choices'):
                text_content = data['choices'][0]['message']['content']
                translated_text = text_content.strip()

                if not translated_text:
                    raise ValueError("LLM returned empty translation text.")
                    
                return translated_text
            
            raise ValueError(f"API returned unparseable response structure.")

        except (requests.exceptions.RequestException, requests.exceptions.HTTPError, ValueError) as e:
            if isinstance(e, requests.exceptions.HTTPError) or isinstance(e, requests.exceptions.RequestException):
                 if attempt < MAX_RETRIES - 1:
                    print(f"API call failed (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2
                 else:
                    print(f"API call failed after {MAX_RETRIES} attempts. Final error: {e}")
                    raise
            else:
                raise

    raise Exception("Translation failed after maximum retries.")

# --- Main Translation Processing Logic ---

def _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False):
    # 1. Rate Limit Check
    if not check_rate_limit(client_ip):
        return jsonify({"error": f"Rate limit exceeded. Try again in {RATE_LIMIT_SECONDS} second(s)."}), 429

    source_language_code = get_source_language(text_to_translate)
    cache_key = f"translation:{text_to_translate}|{target_lang}"
    status_type = "reviewed" if force_refresh else "generated"

    # 2. Cache Check
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
            print(f"Redis cache read error: {e}")
            pass

    # 3. LLM Call Prep
    system_instruction = (
        f"You are a professional, high-quality language translator. Your task is to translate the provided text "
        f"from {source_language_code} to {target_lang}. "
        f"If the text contains slang, abbreviations, or shorthand, make a best effort to translate its intended meaning. "
        f"If the text is truly incomprehensible or nonsensical, provide a literal translation followed by a brief, neutral note in parentheses indicating the uncertainty (e.g., 'Not clear' or 'Abbreviation'). "
        f"IMPORTANT: You MUST ONLY return the translated text and NOTHING else. "
        f"Do not include any introductory phrases, explanations, markdown formatting (like quotes or bolding), or punctuation beyond what is in the translation."
    )
    user_prompt = text_to_translate

    try:
        # 4. LLM Call with Backoff
        translated_text = call_groq_api_with_backoff(system_instruction, user_prompt)
        
        # 5. Prepare Success Response
        response_data = {
            "source_language": source_language_code,
            "target_language": target_lang,
            "translated_text": translated_text,
            "status": status_type
        }
        
        # 6. Cache the result (if successful)
        if r:
            try:
                cache_data = {
                    "source_language": source_language_code,
                    "translated_text": translated_text,
                    "timestamp": time.time(),
                    "model": GROQ_MODEL
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
            except Exception as e:
                print(f"Redis cache write error: {e}")
                pass
        
        return jsonify(response_data), 200

    except ValueError as ve:
        # Catch specific LLM errors (empty translation, unparseable structure)
        print(f"Caught specific LLM response error: {ve}")
        return jsonify({
            "error": "The model struggled to generate a translation. The input may be too short or ambiguous.", 
            "details": str(ve)
        }), 400

    except requests.exceptions.HTTPError as e:
        # Catch Groq API errors (like 401, 403, 404)
        status_code = e.response.status_code
        try:
            error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        except:
            error_details = 'Could not parse error details from Groq.'
        return jsonify({"error": f"Groq API Error ({status_code}): {error_details}"}), status_code

    except Exception as e:
        # Catch any other network/internal server errors
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- API Endpoints ---

@app.route('/translate', methods=['POST'])
def translate():
    client_ip = _get_client_ip()
    
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
    client_ip = _get_client_ip()
    
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

# --- Server Start ---

if __name__ == '__main__':
    print(f"Starting API server (using Groq API with {GROQ_MODEL}).")
    print(f"Access the /translate and /review endpoints via POST request at http://127.0.0.1:5000/translate")
    print("Example Payload: {'text': 'Hello world', 'target_lang': 'French'}")
    app.run(debug=True, host='0.0.0.0', port=5000)

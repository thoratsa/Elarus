import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import redis
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
from functools import wraps

DetectorFactory.seed = 0

REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')
GROQ_MODEL = os.environ.get("GROQ_MODEL")
API_KEY = os.environ.get("GROQ_API_KEY") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 
RATE_LIMIT_SECONDS = 2
MAX_TOKENS_PER_REQUEST = 300
MAX_RETRIES = 5
BASE_DELAY = 0
MAX_TEXT_LENGTH = 2000

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception:
        r = None

app = Flask(__name__, static_folder='public')
CORS(app)

class TranslationError(Exception):
    def __init__(self, message, status_code=400, details=None, error_type=None):
        self.message = message
        self.status_code = status_code
        self.details = details
        self.error_type = error_type or "translation_error"
        super().__init__(self.message)

class RateLimitError(TranslationError):
    def __init__(self, message="Rate limit exceeded", details=None):
        super().__init__(message, 429, details, "rate_limit_error")

class TokenLimitError(TranslationError):
    def __init__(self, message="Token limit exceeded", details=None):
        super().__init__(message, 429, details, "token_limit_error")

class APIKeyError(TranslationError):
    def __init__(self, message="API key not configured", details=None):
        super().__init__(message, 500, details, "api_key_error")

class InputValidationError(TranslationError):
    def __init__(self, message="Invalid input", details=None):
        super().__init__(message, 400, details, "validation_error")

class GroqAPIError(TranslationError):
    def __init__(self, message="Groq API error", details=None, status_code=500):
        super().__init__(message, status_code, details, "groq_api_error")

class CacheError(TranslationError):
    def __init__(self, message="Cache error", details=None):
        super().__init__(message, 500, details, "cache_error")

def validate_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not request.is_json:
                raise InputValidationError(
                    "Request must be JSON format",
                    "Set Content-Type header to application/json"
                )
            
            data = request.get_json()
            if not data:
                raise InputValidationError(
                    "Empty request body",
                    "Request body must contain valid JSON data"
                )
            
            text = data.get('text', '').strip()
            target_lang = data.get('target_lang', '').strip()
            source_lang = data.get('source_lang', '').strip()

            if not text:
                raise InputValidationError(
                    "Text field is required",
                    "Provide text to translate in the 'text' field"
                )
            
            if len(text) > MAX_TEXT_LENGTH:
                raise InputValidationError(
                    f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters",
                    f"Current length: {len(text)} characters. Please shorten your text."
                )
            
            if not target_lang:
                raise InputValidationError(
                    "Target language is required",
                    "Provide target language in the 'target_lang' field"
                )
            
            if not re.match(r'^[A-Za-z\s\-]+$', target_lang):
                raise InputValidationError(
                    "Invalid target language format",
                    "Target language can only contain letters, spaces, and hyphens"
                )
            
            if source_lang and not re.match(r'^[A-Za-z\s\-]+$', source_lang):
                raise InputValidationError(
                    "Invalid source language format",
                    "Source language can only contain letters, spaces, and hyphens"
                )
                
            return f(*args, **kwargs)
            
        except InputValidationError as e:
            return jsonify({
                "error": e.message,
                "details": e.details,
                "error_type": e.error_type,
                "status_code": e.status_code,
                "timestamp": time.time()
            }), e.status_code
        except TranslationError as e:
            return jsonify({
                "error": e.message,
                "details": e.details,
                "error_type": e.error_type,
                "status_code": e.status_code,
                "timestamp": time.time()
            }), e.status_code
        except Exception as e:
            return jsonify({
                "error": "Invalid request format",
                "details": f"Failed to parse request: {str(e)}",
                "error_type": "request_parse_error",
                "status_code": 400,
                "timestamp": time.time()
            }), 400
    return decorated_function

def check_rate_limit(client_id):
    if not r:
        return True, None
    
    current_time = time.time()
    rate_limit_key = f"rate_limit:{client_id}"
    token_limit_key = f"token_limit:{client_id}"
    
    try:
        last_request_time = r.get(rate_limit_key)
        if last_request_time:
            time_since_last = current_time - float(last_request_time)
            if time_since_last < RATE_LIMIT_SECONDS:
                wait_time = RATE_LIMIT_SECONDS - time_since_last
                return False, f"Wait {wait_time:.1f} seconds before next request"
        
        token_count = r.get(token_limit_key)
        if token_count and int(token_count) >= MAX_TOKENS_PER_REQUEST:
            return False, f"Token limit reached ({token_count}/{MAX_TOKENS_PER_REQUEST})"
        
        r.set(rate_limit_key, current_time, ex=RATE_LIMIT_SECONDS * 2)
        return True, None
        
    except Exception as e:
        return True, f"Rate limit check failed: {str(e)}"

def update_token_count(client_id, tokens_used):
    if not r:
        return False
    
    try:
        token_limit_key = f"token_limit:{client_id}"
        current_tokens = r.get(token_limit_key)
        if current_tokens:
            new_total = int(current_tokens) + tokens_used
            r.set(token_limit_key, new_total, ex=86400)
        else:
            r.set(token_limit_key, tokens_used, ex=86400)
        return True
    except Exception as e:
        return False

def get_source_language(text):
    try:
        detected = detect(text)
        return detected.upper() if detected else "UNKNOWN"
    except LangDetectException as e:
        return "UNKNOWN"
    except Exception as e:
        return "UNKNOWN"

def _get_client_ip():
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.remote_addr

def call_groq_api_with_backoff(system_instruction, user_prompt):
    current_delay = BASE_DELAY
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_PER_REQUEST
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()

            data = response.json()
            
            if not data or not data.get('choices'):
                raise GroqAPIError(
                    "Invalid response format from Groq API",
                    "No choices returned in response",
                    502
                )
            
            text_content = data['choices'][0]['message']['content']
            if not text_content:
                raise GroqAPIError(
                    "Empty translation response",
                    "Groq API returned empty content",
                    502
                )
                
            translated_text = text_content.strip()
            if not translated_text:
                raise GroqAPIError(
                    "Empty translation result",
                    "Translation resulted in empty text",
                    502
                )
                    
            return translated_text

        except requests.exceptions.Timeout:
            last_exception = GroqAPIError(
                "Groq API timeout",
                f"Request timed out after 30 seconds (attempt {attempt + 1}/{MAX_RETRIES})",
                504
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown API error')
            except:
                error_msg = str(e)
            
            if status_code == 401:
                last_exception = APIKeyError(
                    "Invalid Groq API key",
                    "Check your GROQ_API_KEY environment variable"
                )
            elif status_code == 429:
                last_exception = GroqAPIError(
                    "Groq API rate limit exceeded",
                    f"Too many requests to Groq API: {error_msg}",
                    429
                )
            else:
                last_exception = GroqAPIError(
                    f"Groq API error (HTTP {status_code})",
                    error_msg,
                    status_code
                )
        except requests.exceptions.RequestException as e:
            last_exception = GroqAPIError(
                "Network error connecting to Groq API",
                f"Request failed: {str(e)}",
                503
            )
        except ValueError as e:
            last_exception = GroqAPIError(
                "Invalid response from Groq API",
                f"Failed to parse response: {str(e)}",
                502
            )
        except Exception as e:
            last_exception = GroqAPIError(
                "Unexpected error calling Groq API",
                f"Unexpected error: {str(e)}",
                500
            )

        if attempt < MAX_RETRIES - 1:
            time.sleep(current_delay)
            current_delay *= 2
        else:
            raise last_exception

    raise GroqAPIError(
        "Translation failed after maximum retries",
        f"Last error: {str(last_exception)}",
        500
    )

def _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False, source_lang_override=None):
    rate_ok, rate_message = check_rate_limit(client_ip)
    if not rate_ok:
        raise RateLimitError(
            f"Rate limit exceeded for IP {client_ip}",
            rate_message
        )

    try:
        source_language_code = source_lang_override.upper() if source_lang_override else get_source_language(text_to_translate)
    except Exception as e:
        source_language_code = "UNKNOWN"

    cache_key = f"translation:{text_to_translate}|{target_lang}"
    status_type = "regenerated" if force_refresh else "generated"

    if not force_refresh and r:
        try:
            cached_result_json = r.get(cache_key)
            if cached_result_json:
                cached_result = json.loads(cached_result_json)
                return {
                    "source_language": cached_result['source_language'],
                    "target_language": target_lang,
                    "translated_text": cached_result['translated_text'],
                    "status": "cached",
                    "cache_timestamp": cached_result.get('timestamp')
                }
        except Exception as e:
            pass

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
        translated_text = call_groq_api_with_backoff(system_instruction, user_prompt)
        
        tokens_used = len(text_to_translate.split()) + len(translated_text.split())
        token_updated = update_token_count(client_ip, tokens_used)
        
        response_data = {
            "source_language": source_language_code,
            "target_language": target_lang,
            "translated_text": translated_text,
            "status": status_type,
            "tokens_used": tokens_used,
            "cache_available": False
        }
        
        if r:
            try:
                cache_data = {
                    "source_language": source_language_code,
                    "translated_text": translated_text,
                    "timestamp": time.time(),
                    "model": GROQ_MODEL,
                    "tokens_used": tokens_used
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
                response_data["cache_available"] = True
            except Exception as e:
                response_data["cache_available"] = False
        
        return response_data

    except Exception as e:
        if isinstance(e, TranslationError):
            raise e
        else:
            raise TranslationError(
                "Translation processing failed",
                f"Unexpected error during translation: {str(e)}",
                500,
                "processing_error"
            )

@app.errorhandler(TranslationError)
def handle_translation_error(error):
    response = jsonify({
        "error": error.message,
        "details": error.details,
        "error_type": error.error_type,
        "status_code": error.status_code,
        "timestamp": time.time()
    })
    response.status_code = error.status_code
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "details": "The requested API endpoint does not exist",
        "error_type": "not_found_error",
        "status_code": 404,
        "timestamp": time.time()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "details": "The HTTP method is not supported for this endpoint",
        "error_type": "method_error",
        "status_code": 405,
        "timestamp": time.time()
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "Internal server error",
        "details": "An unexpected error occurred on the server",
        "error_type": "internal_error",
        "status_code": 500,
        "timestamp": time.time()
    }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    redis_status = "unknown"
    if r:
        try:
            r.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
    else:
        redis_status = "not_configured"
    
    health_status = {
        "status": "healthy",
        "redis": redis_status,
        "groq_api": "configured" if API_KEY else "not_configured",
        "limits": {
            "max_text_length": MAX_TEXT_LENGTH,
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "max_tokens_per_request": MAX_TOKENS_PER_REQUEST
        },
        "timestamp": time.time()
    }
    return jsonify(health_status), 200

@app.route('/api/translate', methods=['POST'])
@validate_input
def translate():
    client_ip = _get_client_ip()
    
    if not API_KEY:
        raise APIKeyError()

    data = request.get_json()
    text_to_translate = data.get('text', '').strip()
    target_lang = data.get('target_lang', '').strip()
    source_lang = data.get('source_lang', '').strip()

    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False, source_lang_override=source_lang or None)
    return jsonify(result), 200

@app.route('/api/retranslate', methods=['POST'])
@validate_input
def retranslate():
    client_ip = _get_client_ip()
    
    if not API_KEY:
        raise APIKeyError()

    data = request.get_json()
    text_to_translate = data.get('text', '').strip()
    target_lang = data.get('target_lang', '').strip()
    source_lang = data.get('source_lang', '').strip()

    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=True, source_lang_override=source_lang or None)
    return jsonify(result), 200

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('public', path)):
        return send_from_directory('public', path)
    abort(404)

@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

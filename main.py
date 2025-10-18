import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import redis
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
from functools import wraps

DetectorFactory.seed = 0

REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')
GROQ_MODEL = "openai/gpt-oss-120b"
API_KEY = os.environ.get("GROQ_API_KEY", "") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 
RATE_LIMIT_SECONDS = 1
MAX_RETRIES = 5
BASE_DELAY = 1
MAX_TEXT_LENGTH = 2000

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception:
        r = None

app = Flask(__name__)
CORS(app)

class TranslationError(Exception):
    def __init__(self, message, status_code=400, details=None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

class RateLimitError(TranslationError):
    def __init__(self, message="Rate limit exceeded", details=None):
        super().__init__(message, 429, details)

class APIKeyError(TranslationError):
    def __init__(self, message="API key not configured", details=None):
        super().__init__(message, 500, details)

class InputValidationError(TranslationError):
    def __init__(self, message="Invalid input", details=None):
        super().__init__(message, 400, details)

def validate_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            data = request.get_json()
            if not data:
                raise InputValidationError("Request must be JSON")
            
            text = data.get('text', '').strip()
            target_lang = data.get('target_lang', '').strip()
            
            if len(text) > MAX_TEXT_LENGTH:
                raise InputValidationError(
                    f"Text too long. Maximum {MAX_TEXT_LENGTH} characters.",
                    f"Text length: {len(text)} characters"
                )
            
            if not text:
                raise InputValidationError("Text cannot be empty")
                
            if not target_lang:
                raise InputValidationError("Target language cannot be empty")
            
            if not re.match(r'^[A-Za-z\s\-]+$', target_lang):
                raise InputValidationError(
                    "Invalid target language format",
                    "Use only letters, spaces, and hyphens"
                )
                
            return f(*args, **kwargs)
        except InputValidationError as e:
            return jsonify({
                "error": e.message,
                "details": e.details,
                "status_code": e.status_code
            }), e.status_code
        except Exception as e:
            return jsonify({
                "error": "Invalid request format",
                "details": str(e),
                "status_code": 400
            }), 400
    return decorated_function

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
                    time.sleep(current_delay)
                    current_delay *= 2
                 else:
                    raise
            else:
                raise

    raise Exception("Translation failed after maximum retries.")

def _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False):
    if not check_rate_limit(client_ip):
        raise RateLimitError(
            f"Rate limit exceeded. Try again in {RATE_LIMIT_SECONDS} second(s).",
            f"Client IP: {client_ip}"
        )

    source_language_code = get_source_language(text_to_translate)
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
                    "status": "cached"
                }
        except Exception:
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
                    "model": GROQ_MODEL
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
            except Exception:
                pass
        
        return response_data

    except ValueError as ve:
        raise TranslationError(
            "The model struggled to generate a translation. The input may be too short or ambiguous.",
            details=str(ve)
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        except:
            error_details = 'Could not parse error details from Groq.'
        raise TranslationError(
            f"Groq API Error ({status_code})",
            status_code=status_code,
            details=error_details
        )

    except Exception as e:
        raise TranslationError(
            "An internal server error occurred",
            status_code=500,
            details=str(e)
        )

@app.errorhandler(TranslationError)
def handle_translation_error(error):
    response = jsonify({
        "error": error.message,
        "details": error.details,
        "status_code": error.status_code
    })
    response.status_code = error.status_code
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "redis": "connected" if r and r.ping() else "disconnected",
        "groq_api": "configured" if API_KEY else "not_configured",
        "max_text_length": MAX_TEXT_LENGTH,
        "rate_limit": f"{RATE_LIMIT_SECONDS} second(s)",
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
    
    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False)
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
    
    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=True)
    return jsonify(result), 200

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Translation API Playground</title>
        <style>
            *{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}.container{max-width:800px;margin:0 auto;background:white;border-radius:15px;box-shadow:0 20px 40px rgba(0,0,0,0.1);overflow:hidden}.header{background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%);color:white;padding:30px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{opacity:0.9;font-size:1.1em}.content{padding:30px}.form-group{margin-bottom:20px}label{display:block;margin-bottom:8px;font-weight:600;color:#2c3e50}textarea,input,select{width:100%;padding:12px;border:2px solid #e1e8ed;border-radius:8px;font-size:16px;transition:border-color 0.3s ease}textarea:focus,input:focus,select:focus{outline:none;border-color:#3498db}textarea{height:120px;resize:vertical}.button-group{display:flex;gap:10px;margin-bottom:20px}button{flex:1;padding:15px;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;transition:all 0.3s ease}.btn-translate{background:linear-gradient(135deg,#27ae60 0%,#2ecc71 100%);color:white}.btn-retranslate{background:linear-gradient(135deg,#e67e22 0%,#f39c12 100%);color:white}button:hover{transform:translateY(-2px);box-shadow:0 5px 15px rgba(0,0,0,0.2)}button:disabled{opacity:0.6;cursor:not-allowed;transform:none}.result{background:#f8f9fa;border-radius:8px;padding:20px;margin-top:20px;border-left:4px solid #3498db}.result h3{color:#2c3e50;margin-bottom:10px}.result-content{white-space:pre-wrap;line-height:1.6}.status{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:10px}.status-cached{background:#d4edda;color:#155724}.status-generated{background:#cce7ff;color:#004085}.status-regenerated{background:#fff3cd;color:#856404}.error{background:#f8d7da;color:#721c24;border-left-color:#dc3545}.loading{text-align:center;padding:20px;color:#6c757d}.api-info{background:#e8f4fd;border-radius:8px;padding:15px;margin-top:20px;font-size:14px}.api-info code{background:#2c3e50;color:white;padding:2px 6px;border-radius:4px;font-family:'Courier New',monospace}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Translation API Playground</h1>
                <p>Test the translation API in real-time with this interactive playground</p>
            </div>
            <div class="content">
                <div class="form-group">
                    <label for="text">Text to Translate (Max 2000 characters):</label>
                    <textarea id="text" placeholder="Enter text to translate..." maxlength="2000">Hello! How are you today? This is a test of the translation API.</textarea>
                    <div style="text-align: right; font-size: 12px; color: #6c757d; margin-top: 5px;">
                        <span id="charCount">0</span>/2000 characters
                    </div>
                </div>
                <div class="form-group">
                    <label for="targetLang">Target Language:</label>
                    <input type="text" id="targetLang" placeholder="e.g., Spanish, French, German, Japanese..." value="Spanish">
                </div>
                <div class="button-group">
                    <button class="btn-translate" onclick="translateText()" id="translateBtn">
                        Translate
                    </button>
                    <button class="btn-retranslate" onclick="retranslateText()" id="retranslateBtn">
                        Retranslate (Force Refresh)
                    </button>
                </div>
                <div id="result"></div>
                <div class="api-info">
                    <h4>API Endpoints:</h4>
                    <p><strong>POST</strong> <code>/api/translate</code> - Standard translation with caching</p>
                    <p><strong>POST</strong> <code>/api/retranslate</code> - Force fresh translation (bypass cache)</p>
                    <p><strong>GET</strong> <code>/api/health</code> - Health check and system status</p>
                    <p><strong>Base URL:</strong> <code>https://elarus.vercel.app</code></p>
                </div>
            </div>
        </div>

        <script>
            const baseUrl = 'https://elarus.vercel.app';
            let currentRequest = null;
            
            const textArea = document.getElementById('text');
            const charCount = document.getElementById('charCount');
            
            textArea.addEventListener('input', function() {
                charCount.textContent = this.value.length;
            });
            
            charCount.textContent = textArea.value.length;
            
            function setLoading(isLoading) {
                const translateBtn = document.getElementById('translateBtn');
                const retranslateBtn = document.getElementById('retranslateBtn');
                
                translateBtn.disabled = isLoading;
                retranslateBtn.disabled = isLoading;
                
                if (isLoading) {
                    translateBtn.innerHTML = 'Translating...';
                    retranslateBtn.innerHTML = 'Retranslating...';
                } else {
                    translateBtn.innerHTML = 'Translate';
                    retranslateBtn.innerHTML = 'Retranslate (Force Refresh)';
                }
            }
            
            function showResult(content, isError = false) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = content;
                if (isError) {
                    resultDiv.classList.add('error');
                } else {
                    resultDiv.classList.remove('error');
                }
            }
            
            async function makeRequest(endpoint) {
                if (currentRequest) {
                    currentRequest.abort();
                }
                
                const text = document.getElementById('text').value.trim();
                const targetLang = document.getElementById('targetLang').value.trim();
                
                if (!text) {
                    showResult('<div class="error"><h3>Error</h3><div class="result-content">Please enter text to translate</div></div>', true);
                    return;
                }
                
                if (!targetLang) {
                    showResult('<div class="error"><h3>Error</h3><div class="result-content">Please enter target language</div></div>', true);
                    return;
                }
                
                if (text.length > 2000) {
                    showResult('<div class="error"><h3>Error</h3><div class="result-content">Text exceeds maximum length of 2000 characters</div></div>', true);
                    return;
                }
                
                setLoading(true);
                showResult('<div class="loading">Processing translation...</div>');
                
                const controller = new AbortController();
                currentRequest = controller;
                
                try {
                    const response = await fetch(`${baseUrl}${endpoint}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            target_lang: targetLang
                        }),
                        signal: controller.signal
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || `HTTP error! status: ${response.status}`);
                    }
                    
                    const statusClass = `status-${data.status}`;
                    const statusText = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                    
                    showResult(`
                        <div class="result">
                            <span class="status ${statusClass}">${statusText}</span>
                            <h3>Translation Result</h3>
                            <div class="result-content">
                                <strong>Source:</strong> ${data.source_language} -> <strong>Target:</strong> ${data.target_language}
                                <br><br>
                                <strong>Translated Text:</strong>
                                <br>
                                ${data.translated_text}
                            </div>
                        </div>
                    `);
                    
                } catch (error) {
                    if (error.name === 'AbortError') {
                        return;
                    }
                    showResult(`
                        <div class="result error">
                            <h3>Error</h3>
                            <div class="result-content">
                                ${error.message}
                                ${error.details ? '<br><br><strong>Details:</strong> ' + error.details : ''}
                            </div>
                        </div>
                    `, true);
                } finally {
                    setLoading(false);
                    currentRequest = null;
                }
            }
            
            function translateText() {
                makeRequest('/api/translate');
            }
            
            function retranslateText() {
                makeRequest('/api/retranslate');
            }
            
            document.getElementById('targetLang').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    translateText();
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

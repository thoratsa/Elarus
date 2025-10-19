import os
import json
import requests
from flask import Flask, request, jsonify
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
GROQ_TIMEOUT = 30

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)
        r.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        r = None
else:
    print("Redis URL not provided, caching disabled")

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
    except Exception as e:
        print(f"Rate limit check failed: {e}")
        return True

def get_source_language(text):
    try:
        lang = detect(text).upper()
        return lang if lang != "UN" else "Unknown"
    except LangDetectException as e:
        print(f"Language detection failed: {e}")
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
        "max_tokens": 1000
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=GROQ_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            
            if not data or not data.get('choices'):
                raise ValueError("API returned empty or invalid response")
            
            text_content = data['choices'][0]['message']['content']
            translated_text = text_content.strip()

            if not translated_text:
                raise ValueError("LLM returned empty translation text")
                
            return translated_text

        except (requests.exceptions.RequestException, requests.exceptions.HTTPError, ValueError) as e:
            if isinstance(e, (requests.exceptions.HTTPError, requests.exceptions.RequestException)):
                 if attempt < MAX_RETRIES - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                 else:
                    raise
            else:
                raise

    raise Exception("Translation failed after maximum retries")

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
        except Exception as e:
            print(f"Cache read error: {e}")

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
            except Exception as e:
                print(f"Cache write error: {e}")
        
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "details": "The requested endpoint does not exist",
        "status_code": 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "details": "An unexpected error occurred",
        "status_code": 500
    }), 500

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

HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elarus - Translation API</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
        }

        .header {
            background: #000;
            color: white;
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid #333;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
            opacity: 0.9;
        }

        .tagline {
            opacity: 0.7;
            font-size: 1rem;
        }

        .content {
            flex: 1;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }

        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            align-items: start;
        }

        .form-group {
            margin-bottom: 2rem;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #ccc;
            font-size: 0.9rem;
        }

        textarea, input {
            width: 100%;
            padding: 1rem;
            border: 1px solid #333;
            border-radius: 8px;
            font-size: 1rem;
            font-family: inherit;
            background: #2d2d2d;
            color: #e0e0e0;
            transition: all 0.2s ease;
        }

        textarea:focus, input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
        }

        textarea {
            height: 140px;
            resize: vertical;
            line-height: 1.5;
        }

        .char-counter {
            text-align: right;
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.25rem;
        }

        .button-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .btn {
            flex: 1;
            padding: 1rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            background: #5a6fd8;
            transform: translateY(-1px);
        }

        .btn-secondary {
            background: #555;
            color: white;
        }

        .btn-secondary:hover:not(:disabled) {
            background: #666;
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .result {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid #333;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
        }

        .result h3 {
            color: #fff;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .result-content {
            white-space: pre-wrap;
            line-height: 1.5;
            color: #e0e0e0;
        }

        .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .status-cached {
            background: #1e3a28;
            color: #4ade80;
            border: 1px solid #2d4a3a;
        }

        .status-generated {
            background: #1e3a5f;
            color: #60a5fa;
            border: 1px solid #2d4a7a;
        }

        .status-regenerated {
            background: #5c4a1e;
            color: #fbbf24;
            border: 1px solid #7a5a2d;
        }

        .error {
            background: #3a1e1e;
            border-color: #7a2d2d;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #888;
            font-style: italic;
        }

        .api-info {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 1px solid #333;
            grid-column: 1 / -1;
        }

        .api-info h3 {
            color: #fff;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .api-info p {
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #ccc;
        }

        .api-info code {
            background: #000;
            color: #667eea;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.8rem;
        }

        .json-key {
            color: #f78c6c;
            font-weight: 600;
        }

        .json-string {
            color: #89ddff;
        }

        .json-number {
            color: #f78c6c;
        }

        .json-boolean {
            color: #c792ea;
        }

        @media (max-width: 767px) {
            .content {
                padding: 1rem;
            }
            
            .header {
                padding: 1.5rem 1rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Elarus</h1>
            <div class="subtitle">Translation API Playground</div>
            <div class="tagline">High-performance translation microservice powered by AI</div>
        </header>
        
        <main class="content">
            <div class="content-grid">
                <div class="input-section">
                    <div class="form-group">
                        <label for="text">Text to Translate (Max 2000 characters):</label>
                        <textarea id="text" placeholder="Enter text to translate..." maxlength="2000">Hello! How are you today? This is a test of the Elarus translation API.</textarea>
                        <div class="char-counter">
                            <span id="charCount">0</span>/2000 characters
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="targetLang">Target Language:</label>
                        <input type="text" id="targetLang" placeholder="e.g., Spanish, French, German, Japanese..." value="Spanish">
                    </div>
                    
                    <div class="button-group">
                        <button class="btn btn-primary" onclick="translateText()" id="translateBtn">
                            Translate
                        </button>
                        <button class="btn btn-secondary" onclick="retranslateText()" id="retranslateBtn">
                            Retranslate
                        </button>
                    </div>
                </div>
                
                <div class="output-section">
                    <div id="result"></div>
                </div>
                
                <div class="api-info">
                    <h3>Elarus API Endpoints</h3>
                    <p><strong>POST</strong> <code>/api/translate</code> - Standard translation with intelligent caching</p>
                    <p><strong>POST</strong> <code>/api/retranslate</code> - Force fresh translation (bypass cache)</p>
                    <p><strong>GET</strong> <code>/api/health</code> - System status and health check</p>
                    <p><strong>Base URL:</strong> <code>https://elarus.vercel.app</code></p>
                </div>
            </div>
        </main>
    </div>

    <script>
        const baseUrl = window.location.origin;
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
                retranslateBtn.innerHTML = 'Retranslate';
            }
        }

        function formatJSON(data) {
            return JSON.stringify(data, null, 2)
                .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
                .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
                .replace(/: (\d+)/g, ': <span class="json-number">$1</span>')
                .replace(/: (true|false)/g, ': <span class="json-boolean">$1</span>');
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
                const formattedJSON = formatJSON(data);
                
                showResult(`
                    <div class="result">
                        <span class="status ${statusClass}">${statusText}</span>
                        <h3>API Response</h3>
                        <div class="result-content">${formattedJSON}</div>
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
'''

@app.route('/')
def serve_index():
    return HTML_CONTENT

if __name__ == '__main__':
    print(f"Starting Elarus Translation API (Groq model: {GROQ_MODEL})")
    print(f"Playground available at: http://localhost:5000")
    print(f"API endpoints available at: /api/translate, /api/retranslate, /api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)

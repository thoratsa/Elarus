# Elarus

Elarus provides high-performance, real-time translation services powered by Groq's LPU infrastructure. It offers a simple JSON API for AI-powered translations with smart caching and rate limiting.

## API endpoints

The service provides three main endpoints for translations and system status.

**Live Playground**: [https://elarus.vercel.app](https://elarus.vercel.app)

### Translate
```http
POST /api/translate
```
Standard translation with intelligent caching.

### Retranslate
```http
POST /api/retranslate
```
Force fresh translation, bypassing cache.

### Health Check
```http
GET /api/health
```
Check API status and configuration.

## Quick Start

### Using the API

Translate text from English to Spanish:

```bash
curl -X POST https://elarus.vercel.app/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?", "target_lang": "Spanish"}'
```

Response:
```json
{
  "source_language": "EN",
  "target_language": "Spanish",
  "translated_text": "Hola, ¿cómo estás hoy?",
  "status": "generated"
}
```

### Force Fresh Translation

Get a new translation, bypassing cache:

```bash
curl -X POST https://elarus.vercel.app/api/retranslate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?", "target_lang": "French"}'
```

### Check Service Status

```bash
curl https://elarus.vercel.app/api/health
```

## Features

- **AI-Powered Translations**: Powered by Groq's GPT-OSS-120B model
- **Smart Caching**: Redis-based caching for improved performance
- **Rate Limiting**: 1 request per second per IP address
- **Auto Language Detection**: Automatically detects source language
- **Modern Dark UI**: Full-screen playground with dark theme
- **JSON API**: Clean, consistent JSON responses

## Usage Examples

### Python

```python
import requests

def translate_text(text, target_lang):
    response = requests.post(
        'https://elarus.vercel.app/api/translate',
        json={'text': text, 'target_lang': target_lang}
    )
    return response.json()

# Usage
result = translate_text('Good morning', 'Japanese')
print(f"Translation: {result['translated_text']}")
```

### JavaScript

```javascript
async function translateText(text, targetLang) {
  const response = await fetch('https://elarus.vercel.app/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, target_lang: targetLang })
  });
  return await response.json();
}

// Usage
translateText('Thank you', 'German')
  .then(result => console.log(result.translated_text));
```

## Supported Languages

Elarus supports all languages available in Groq's GPT-OSS-120B model, including but not limited to:

- English, Spanish, French, German, Italian, Portuguese
- Chinese, Japanese, Korean, Arabic, Russian
- And 100+ other languages

## Response Format

### Success Response
```json
{
  "source_language": "EN",
  "target_language": "Spanish",
  "translated_text": "Texto traducido",
  "status": "cached"
}
```

### Error Response
```json
{
  "error": "Rate limit exceeded",
  "details": "Try again in 1 second(s)",
  "status_code": 429
}
```

## Rate Limits

- **1 request per second** per IP address
- Automatic rate limiting with Redis
- 429 status code when limit exceeded

## Text Limits

- Maximum text length: **2000 characters**
- Input validation for text and language format

## Deployment

The API is deployed on Vercel with serverless architecture. Key files:

- `app.py` - Main Flask application
- `index.html` - Playground interface
- `style.css` - Dark theme styling
- `script.js` - Frontend functionality

## Support

For issues and questions:
1. Test with the [playground](https://elarus.vercel.app)
2. Check the health endpoint for service status
3. Ensure you're within rate limits

## License

This project is licensed under the MIT License, check [LICENSE](LICENSE) for more information.

---

**Elarus** - Fast, reliable AI translation API.

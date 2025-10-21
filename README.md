[![license](https://img.shields.io/github/license/iamtraction/google-translate.svg)](LICENSE)
![Python Badge](https://img.shields.io/badge/Made%20with-Python-blue)
# Elarus

Elarus provides high-performance, real-time translation services powered by Groq's LPU infrastructure. It offers a simple JSON API for AI-powered translations with smart caching and rate limiting.

## Quick Start

The API is available at `https://elarus.vercel.app/api` with three main endpoints:

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/translate` | POST | Standard translation with intelligent caching |
| `/retranslate` | POST | Force fresh translation (bypass cache) |
| `/health` | GET | System status and health check |

### Request Format
All translation endpoints accept JSON with:
```json
{
  "text": "Text to translate",
  "target_lang": "Target language",
  "source_lang": "Source language (optional)"
}
```

### Response Format
Successful responses return:
```json
{
  "source_language": "EN",
  "target_language": "Spanish", 
  "translated_text": "Texto traducido",
  "status": "cached"
}
```

## Usage Examples

### Bash (curl)
```bash
# Basic translation
curl -X POST https://elarus.vercel.app/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_lang": "Spanish"}'

# Force fresh translation
curl -X POST https://elarus.vercel.app/api/retranslate \
  -H "Content-Type": application/json" \
  -d '{"text": "Hello world", "target_lang": "French"}'

# Health check
curl https://elarus.vercel.app/api/health
```

### Python
```python
import requests

def translate_text(text, target_lang, force_refresh=False):
    endpoint = "/api/retranslate" if force_refresh else "/api/translate"
    response = requests.post(
        f"https://elarus.vercel.app{endpoint}",
        json={"text": text, "target_lang": target_lang}
    )
    return response.json()

# Usage examples
result = translate_text("Good morning", "Japanese")
print(f"Translation: {result['translated_text']}")

# Force fresh translation
fresh_result = translate_text("Thank you", "German", force_refresh=True)
print(f"Status: {fresh_result['status']}")
```

### JavaScript
```javascript
class ElarusClient {
  constructor(baseUrl = 'https://elarus.vercel.app') {
    this.baseUrl = baseUrl;
  }

  async translate(text, targetLang, forceRefresh = false) {
    const endpoint = forceRefresh ? '/api/retranslate' : '/api/translate';
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, target_lang: targetLang })
    });
    return await response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/api/health`);
    return await response.json();
  }
}

// Usage
const client = new ElarusClient();
client.translate('Hello world', 'Spanish')
  .then(result => console.log(result.translated_text));
```

### Java
```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.google.gson.Gson;

public class ElarusClient {
    private static final String BASE_URL = "https://elarus.vercel.app";
    private final HttpClient client = HttpClient.newHttpClient();
    private final Gson gson = new Gson();

    public TranslationResult translate(String text, String targetLang, boolean forceRefresh) throws Exception {
        String endpoint = forceRefresh ? "/api/retranslate" : "/api/translate";
        String json = gson.toJson(new TranslationRequest(text, targetLang));
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + endpoint))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
                
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return gson.fromJson(response.body(), TranslationResult.class);
    }
    
    // Helper classes
    static class TranslationRequest {
        String text;
        String target_lang;
        TranslationRequest(String text, String targetLang) {
            this.text = text;
            this.target_lang = targetLang;
        }
    }
    
    static class TranslationResult {
        String source_language;
        String target_language;
        String translated_text;
        String status;
    }
}
```

### Go
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type TranslationRequest struct {
    Text      string `json:"text"`
    TargetLang string `json:"target_lang"`
}

type TranslationResult struct {
    SourceLanguage string `json:"source_language"`
    TargetLanguage string `json:"target_language"`
    TranslatedText string `json:"translated_text"`
    Status         string `json:"status"`
}

func Translate(text, targetLang string, forceRefresh bool) (*TranslationResult, error) {
    endpoint := "/api/translate"
    if forceRefresh {
        endpoint = "/api/retranslate"
    }
    
    reqBody := TranslationRequest{Text: text, TargetLang: targetLang}
    jsonData, _ := json.Marshal(reqBody)
    
    resp, err := http.Post("https://elarus.vercel.app"+endpoint, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    var result TranslationResult
    err = json.NewDecoder(resp.Body).Decode(&result)
    return &result, err
}

// Usage
func main() {
    result, err := Translate("Hello world", "Spanish", false)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Translation: %s\n", result.TranslatedText)
}
```

## Features

- **AI-Powered**: Uses Groq's GPT-OSS-120B model for high-quality translations
- **Smart Caching**: Redis-based caching reduces latency for repeated and common translations
- **Auto Language Detection**: Automatically identifies source language
- **JSON API**: Consistent RESTful API with detailed error responses
- **Multi-language Support**: 100+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic, and many more
- **Fast Response Times**: Typically responds within milliseconds to 5 seconds depending on text length and cache status

## Limits & Performance

- **Rate Limit**: 1 request per 2 seconds for every IP address
- **Token Limit**: Maximum 300 tokens per request
- **Text Length**: Maximum 2000 characters per request
- **Response Time**: Typically between milliseconds and 5 seconds depending on text length and cache status
- **Cached Responses**: Near-instant response for repeated translations

## Error Handling

All errors return consistent JSON format:
```json
{
  "error": "Error description",
  "details": "Additional context", 
  "status_code": 400
}
```

Common error codes:
- `400` - Invalid request (missing fields, invalid format)
- `429` - Rate limit exceeded
- `500` - Internal server error

## Live Playground

Test the API with our interactive playground: [https://elarus.vercel.app](https://elarus.vercel.app)

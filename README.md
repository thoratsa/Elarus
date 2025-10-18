# Elarus API: Translation Microservice API

Elarus is a **powerful, serverless microservice** designed for speed and efficiency. Built on Flask and powered by the Groq API, Elarus delivers high-quality, real-time translations with astonishingly low latency. It is the perfect backend for any application requiring instant, reliable language support.

---

## API Usage: Quick Start

The easiest way to use Elarus is by sending a **POST** request to the live, deployed endpoint.

### API Endpoint Details

| Method | URL | 
 | ----- | ----- | 
| **POST** | `https://elarus.vercel.app/translate` | 

### Request Payload (The Data to Send)

Your request must include a JSON body with the following two keys:

| Parameter | Type | Required | Description |
| ----- | ----- | ----- | ----- | 
| `text` | string | Yes | The source text you want to translate. | 
| `target_lang` | string | Yes | The desired language for the translation (e.g., "Spanish", "German", "Japanese"). | 

### Test the API (Using `curl`)

Use this command in any terminal to test the API and see the result:

```bash
curl -X POST [https://elarus.vercel.app/translate](https://elarus.vercel.app/translate) \
  -H "Content-Type: application/json" \
  -d '{"text": "The API is ready for integration.", "target_lang": "Spanish"}'
```

### Example Success Response

The API returns a JSON object containing the translated text and language details. The `source_language` is accurately detected, and a `status` field indicates if the result came from the cache or was newly generated:

```json
{
  "source_language": "English", 
  "target_language": "Spanish",
  "translated_text": "La API está lista para la integración.",
  "status": "cached" 
}
```

## Key Features & How It Works (Performance & Quality)

Elarus is optimized for speed, security, and efficiency:

1.  **Integrated Rate Limiting:** The API implements a strict **1 request per second (RPS) limit per IP address** using the Redis cache. This prevents abuse and manages unexpected costs associated with the Groq API. Hitting this limit will return a `429 Too Many Requests` status.
    
2.  **Accurate Source Language:** The API explicitly instructs the LLM to identify the original text's language, providing an accurate value (e.g., `"English"`, `"French"`) in the `source_language` field of the response.
    
3.  **Blazing Speed via Groq:** The translation is handled by Groq's specialized Language Processing Unit (LPU) infrastructure, resulting in near-instantaneous API responses.
    
4.  **External Caching Layer (Redis):** The API is connected to a dedicated, external **Upstash Redis** instance. This allows the service to check the cache for an existing translation before calling the expensive Groq API. This ensures reliable, permanent persistence across all serverless function calls.
    
5.  **Simple Integration (CORS):** The service is configured with **Cross-Origin Resource Sharing (CORS)** enabled, meaning any client can easily integrate with the API without security restrictions.
    

----------

Thanks for checking out Elarus! We hope this makes integrating powerful, fast translation into your project simple and easy.

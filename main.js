const express = require('express');
const cors = require('cors');
const redis = require('redis');
const fetch = require('node-fetch');
const franc = require('franc');
const path = require('path');
const rateLimit = require('express-rate-limit');

const app = express();

// Configuration
const GROQ_MODEL = "openai/gpt-oss-120b";
const API_KEY = process.env.GROQ_API_KEY || "";
const API_URL = "https://api.groq.com/openai/v1/chat/completions";
const CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7;
const MAX_RETRIES = 5;
const BASE_DELAY = 1000;
const MAX_TEXT_LENGTH = 2000;

// Redis client
let redisClient;
if (process.env.REDIS_URL_REDIS_URL) {
    redisClient = redis.createClient({
        url: process.env.REDIS_URL_REDIS_URL
    });
    
    redisClient.on('error', (err) => {
        console.log('Redis Client Error', err);
        redisClient = null;
    });
    
    redisClient.connect().catch(() => {
        redisClient = null;
    });
}

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Rate limiting
const limiter = rateLimit({
    windowMs: 1000, // 1 second
    max: 1, // 1 request per second
    message: {
        error: 'Rate limit exceeded',
        details: 'Try again in 1 second',
        status_code: 429
    },
    keyGenerator: (req) => {
        return req.headers['x-forwarded-for'] || req.ip;
    }
});

// Custom error class
class TranslationError extends Error {
    constructor(message, statusCode = 400, details = null) {
        super(message);
        this.statusCode = statusCode;
        this.details = details;
    }
}

// Utility functions
function validateInput(data) {
    if (!data) {
        throw new TranslationError('Request must be JSON');
    }
    
    const text = (data.text || '').trim();
    const targetLang = (data.target_lang || '').trim();
    
    if (text.length > MAX_TEXT_LENGTH) {
        throw new TranslationError(
            `Text too long. Maximum ${MAX_TEXT_LENGTH} characters.`,
            400,
            `Text length: ${text.length} characters`
        );
    }
    
    if (!text) {
        throw new TranslationError('Text cannot be empty');
    }
    
    if (!targetLang) {
        throw new TranslationError('Target language cannot be empty');
    }
    
    if (!/^[A-Za-z\s\-]+$/.test(targetLang)) {
        throw new TranslationError(
            'Invalid target language format',
            400,
            'Use only letters, spaces, and hyphens'
        );
    }
    
    return { text, targetLang };
}

function getSourceLanguage(text) {
    try {
        const langCode = franc(text);
        // Convert language code to readable format
        const languageMap = {
            'eng': 'EN',
            'spa': 'ES', 
            'fra': 'FR',
            'deu': 'DE',
            'ita': 'IT',
            'por': 'PT',
            'rus': 'RU',
            'jpn': 'JA',
            'kor': 'KO',
            'zho': 'ZH',
            'ara': 'AR'
        };
        return languageMap[langCode] || langCode.toUpperCase();
    } catch (error) {
        return 'Unknown';
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function callGroqApiWithBackoff(systemInstruction, userPrompt) {
    let currentDelay = BASE_DELAY;
    
    const payload = {
        model: GROQ_MODEL,
        messages: [
            { role: "system", content: systemInstruction },
            { role: "user", content: userPrompt }
        ],
        temperature: 0.2,
    };

    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
    };

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data && data.choices && data.choices.length > 0) {
                const textContent = data.choices[0].message.content;
                const translatedText = textContent.trim();

                if (!translatedText) {
                    throw new Error("LLM returned empty translation text.");
                }
                
                return translatedText;
            }
            
            throw new Error("API returned unparseable response structure.");

        } catch (error) {
            if (attempt < MAX_RETRIES - 1) {
                console.log(`API call failed (Attempt ${attempt + 1}/${MAX_RETRIES}). Error: ${error.message}. Retrying in ${currentDelay}ms...`);
                await sleep(currentDelay);
                currentDelay *= 2;
            } else {
                throw error;
            }
        }
    }

    throw new Error("Translation failed after maximum retries.");
}

async function processTranslation(textToTranslate, targetLang, clientIp, forceRefresh = false) {
    const sourceLanguageCode = getSourceLanguage(textToTranslate);
    const cacheKey = `translation:${textToTranslate}|${targetLang}`;
    const statusType = forceRefresh ? "regenerated" : "generated";

    // Check cache
    if (!forceRefresh && redisClient) {
        try {
            const cachedResultJson = await redisClient.get(cacheKey);
            if (cachedResultJson) {
                const cachedResult = JSON.parse(cachedResultJson);
                return {
                    source_language: cachedResult.source_language,
                    target_language: targetLang,
                    translated_text: cachedResult.translated_text,
                    status: "cached"
                };
            }
        } catch (error) {
            console.log('Redis cache read error:', error);
        }
    }

    const systemInstruction = `
        You are a professional, high-quality language translator. Your task is to translate the provided text 
        from ${sourceLanguageCode} to ${targetLang}. 
        If the text contains slang, abbreviations, or shorthand, make a best effort to translate its intended meaning. 
        If the text is truly incomprehensible or nonsensical, provide a literal translation followed by a brief, neutral note in parentheses indicating the uncertainty (e.g., 'Not clear' or 'Abbreviation'). 
        IMPORTANT: You MUST ONLY return the translated text and NOTHING else. 
        Do not include any introductory phrases, explanations, markdown formatting (like quotes or bolding), or punctuation beyond what is in the translation.
    `;

    try {
        const translatedText = await callGroqApiWithBackoff(systemInstruction, textToTranslate);
        
        const responseData = {
            source_language: sourceLanguageCode,
            target_language: targetLang,
            translated_text: translatedText,
            status: statusType
        };
        
        // Cache the result
        if (redisClient) {
            try {
                const cacheData = {
                    source_language: sourceLanguageCode,
                    translated_text: translatedText,
                    timestamp: Date.now(),
                    model: GROQ_MODEL
                };
                await redisClient.setEx(cacheKey, CACHE_EXPIRY_SECONDS, JSON.stringify(cacheData));
            } catch (error) {
                console.log('Redis cache write error:', error);
            }
        }
        
        return responseData;

    } catch (error) {
        if (error.message.includes('HTTP error')) {
            const statusCode = error.message.match(/status: (\d+)/)?.[1] || 500;
            throw new TranslationError(
                `Groq API Error (${statusCode})`,
                parseInt(statusCode),
                error.message
            );
        } else if (error.message.includes('empty translation')) {
            throw new TranslationError(
                "The model struggled to generate a translation. The input may be too short or ambiguous.",
                400,
                error.message
            );
        } else {
            throw new TranslationError(
                "An internal server error occurred",
                500,
                error.message
            );
        }
    }
}

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof TranslationError) {
        return res.status(error.statusCode).json({
            error: error.message,
            details: error.details,
            status_code: error.statusCode
        });
    }
    
    console.error('Unhandled error:', error);
    res.status(500).json({
        error: 'Internal server error',
        details: error.message,
        status_code: 500
    });
});

// Routes
app.get('/api/health', async (req, res) => {
    let redisStatus = 'disconnected';
    if (redisClient) {
        try {
            await redisClient.ping();
            redisStatus = 'connected';
        } catch (error) {
            redisStatus = 'disconnected';
        }
    }
    
    const healthStatus = {
        status: "healthy",
        redis: redisStatus,
        groq_api: API_KEY ? "configured" : "not_configured",
        max_text_length: MAX_TEXT_LENGTH,
        rate_limit: "1 second(s)",
        timestamp: Date.now()
    };
    
    res.json(healthStatus);
});

app.post('/api/translate', limiter, async (req, res, next) => {
    try {
        if (!API_KEY) {
            throw new TranslationError('API key not configured', 500);
        }

        const { text, targetLang } = validateInput(req.body);
        const clientIp = req.headers['x-forwarded-for'] || req.ip;
        
        const result = await processTranslation(text, targetLang, clientIp, false);
        res.json(result);
    } catch (error) {
        next(error);
    }
});

app.post('/api/retranslate', limiter, async (req, res, next) => {
    try {
        if (!API_KEY) {
            throw new TranslationError('API key not configured', 500);
        }

        const { text, targetLang } = validateInput(req.body);
        const clientIp = req.headers['x-forwarded-for'] || req.ip;
        
        const result = await processTranslation(text, targetLang, clientIp, true);
        res.json(result);
    } catch (error) {
        next(error);
    }
});

// Serve static files
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Elarus API server running on port ${PORT}`);
    console.log(`Access the playground at http://localhost:${PORT}`);
});

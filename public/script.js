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

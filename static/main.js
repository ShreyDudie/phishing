document.getElementById('check-url-btn').addEventListener('click', () => {
    const urlInput = document.getElementById('url-input').value;
    const resultDiv = document.getElementById('url-result');

    if (urlInput.trim() === '') {
        resultDiv.innerHTML = `<div class="alert alert-info" role="alert">Please enter a URL to check.</div>`;
        return;
    }

    resultDiv.innerHTML = `<div class="alert alert-info" role="alert">Checking...</div>`;

    fetch('https://phishing-ntjk.onrender.com/check-url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: urlInput })
    })
   .then(response => response.json())
   .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger" role="alert">Error: ${data.error}</div>`;
            return;
        }

        const classification = data.classification;
        const score = (data.score * 100).toFixed(2);
        let resultClass = '';
        let resultText = '';
        
        if (classification === 'phishing') {
            resultClass = 'warning';
            resultText = `This URL is classified as **Phishing** with a risk score of **${score}%**. Proceed with caution.`;
        } else {
            resultClass = 'safe';
            resultText = `This URL is classified as **Legitimate**. It appears to be safe.`;
        }

        resultDiv.innerHTML = `
            <div class="result-box ${resultClass}">
                <p>${resultText}</p>
            </div>
        `;
    })
   .catch(error => {
        resultDiv.innerHTML = `<div class="alert alert-danger" role="alert">An error occurred while connecting to the server.</div>`;
    });
});

document.getElementById('check-sms-btn').addEventListener('click', () => {
    const smsInput = document.getElementById('sms-input').value;
    const resultDiv = document.getElementById('sms-result');

    if (smsInput.trim() === '') {
        resultDiv.innerHTML = `<div class="alert alert-info" role="alert">Please enter a message to check.</div>`;
        return;
    }

    resultDiv.innerHTML = `<div class="alert alert-info" role="alert">Checking...</div>`;

    fetch('https://phishing-ntjk.onrender.com/check-sms', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: smsInput })
    })
   .then(response => response.json())
   .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<div class="alert alert-danger" role="alert">Error: ${data.error}</div>`;
            return;
        }

        const classification = data.classification;
        const hamConfidence = (data.confidence.ham * 100).toFixed(2);
        const spamConfidence = (data.confidence.spam * 100).toFixed(2);
        
        let resultClass = '';
        let resultText = '';

        if (classification === 'spam') {
            resultClass = 'warning';
            resultText = `This message is classified as **Spam** with a confidence of **${spamConfidence}%**.`;
        } else {
            resultClass = 'safe';
            resultText = `This message is classified as **Legitimate** with a confidence of **${hamConfidence}%**.`;
        }

        resultDiv.innerHTML = `
            <div class="result-box ${resultClass}">
                <p>${resultText}</p>
            </div>
        `;
    })
   .catch(error => {
        resultDiv.innerHTML = `<div class="alert alert-danger" role="alert">An error occurred while connecting to the server.</div>`;
    });
});
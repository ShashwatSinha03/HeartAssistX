// API Base URL (Assuming FastAPI runs locally on 8000)
const API_BASE = 'http://127.0.0.1:8000';
let currentVersion = 'v1'; // Default
let historyChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const versionToggle = document.getElementById('version-toggle');
    const body = document.body;
    
    // Theme Toggle Logic
    themeToggle.addEventListener('change', () => {
        if (themeToggle.checked) {
            body.classList.remove('light-mode');
            body.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
            body.classList.add('light-mode');
        }
    });

    // Version Toggle Logic
    const headerBadge = document.getElementById('header-badge');
    const versionDesc = document.getElementById('version-desc');
    const v2ResultsContainer = document.getElementById('v2-results');

    versionToggle.addEventListener('change', () => {
        if (versionToggle.checked) {
            currentVersion = 'v2';
            headerBadge.textContent = 'V2';
            headerBadge.className = 'badge v2-badge';
            versionDesc.textContent = 'Agentic AI integration. Explainability, Memory, and LLM Recommendations enabled.';
            v2ResultsContainer.classList.remove('hidden');
        } else {
            currentVersion = 'v1';
            headerBadge.textContent = 'V1';
            headerBadge.className = 'badge v1-badge';
            versionDesc.textContent = 'Direct prediction endpoint. No LLM integration.';
            v2ResultsContainer.classList.add('hidden');
        }
        // Hide results on switch
        document.getElementById('results-section').classList.add('hidden');
    });

    // Form Submission
    const riskForm = document.getElementById('risk-form');
    riskForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const inputs = {
            "age": parseFloat(document.getElementById('age').value),
            "sex": parseInt(document.getElementById('sex').value),
            "cp": parseInt(document.getElementById('cp').value),
            "trestbps": parseFloat(document.getElementById('trestbps').value),
            "chol": parseFloat(document.getElementById('chol').value),
            "fbs": parseInt(document.getElementById('fbs').value),
            "restecg": parseInt(document.getElementById('restecg').value),
            "thalach": parseFloat(document.getElementById('thalach').value),
            "exang": parseInt(document.getElementById('exang').value),
            "oldpeak": parseFloat(document.getElementById('oldpeak').value),
            "slope": parseInt(document.getElementById('slope').value),
            "ca": parseInt(document.getElementById('ca').value),
            "thal": parseInt(document.getElementById('thal').value)
        };

        const btn = document.getElementById('predict-btn');
        const spinner = document.getElementById('loading-spinner');
        const resultsSection = document.getElementById('results-section');
        
        btn.disabled = true;
        spinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            if (currentVersion === 'v1') {
                await fetchV1(inputs);
            } else {
                await fetchV2(inputs);
            }
        } catch (error) {
            console.error(error);
            alert("Error communicating with Backend. Make sure FastAPI is running on port 8000.");
        } finally {
            btn.disabled = false;
            spinner.classList.add('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

async function fetchV1(inputs) {
    const response = await fetch(`${API_BASE}/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs)
    });
    if (!response.ok) throw new Error("API Error");
    const data = await response.json();
    
    displayV1Results(data.risk_probability, data.label);
}

async function fetchV2(inputs) {
    const response = await fetch(`${API_BASE}/v2/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs)
    });
    if (!response.ok) throw new Error("API Error");
    const data = await response.json();
    
    displayV1Results(data.prediction.score, data.prediction.label);
    
    // V2 Data
    document.getElementById('v2-results').classList.remove('hidden');
    
    // Top Factors
    const list = document.getElementById('top-factors-list');
    list.innerHTML = '';
    data.top_factors.forEach(f => {
        const li = document.createElement('li');
        li.textContent = `${f.feature.toUpperCase()}: ${f.contribution.toFixed(4)}`;
        list.appendChild(li);
    });

    // LLM Explanation & Recommendation
    document.getElementById('ai-explanation').textContent = data.explanation;
    document.getElementById('ai-recommendation').textContent = data.recommendation;

    // Load History
    await updateHistoryChart();
}

function displayV1Results(prob, label) {
    document.getElementById('results-section').classList.remove('hidden');
    
    const percentage = (prob * 100).toFixed(1) + '%';
    document.getElementById('risk-score').textContent = percentage;

    const riskAlert = document.getElementById('risk-alert');
    const alertTitle = document.getElementById('alert-title');
    const alertText = document.getElementById('alert-text');

    if (label === 'High Risk') {
        riskAlert.className = 'risk-alert alert-high';
        alertTitle.textContent = '🚨 HIGH RISK ALERT';
        alertText.textContent = 'Diagnostic indicators suggest high cardiovascular risk.';
    } else {
        riskAlert.className = 'risk-alert alert-low';
        alertTitle.textContent = '✅ LOW RISK OBSERVED';
        alertText.textContent = 'Clinical metrics align with low risk profiles.';
    }
}

async function updateHistoryChart() {
    const res = await fetch(`${API_BASE}/v2/history`);
    const history = await res.json();
    
    if (historyChartInstance) {
        historyChartInstance.destroy();
    }
    
    const labels = history.map((_, i) => `Entry ${i+1}`);
    const dataPoints = history.map(h => h.risk_score * 100);

    const ctx = document.getElementById('historyChart').getContext('2d');
    historyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Score (%)',
                data: dataPoints,
                borderColor: '#ef4444',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

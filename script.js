let modelParams = null;

// Function to load model parameters
async function loadModel() {
    try {
        const response = await fetch('model_params.json');
        if (!response.ok) throw new Error('Failed to load model_params.json');
        modelParams = await response.json();
        
        // Update UI with model info
        document.getElementById('model-name-text').textContent = modelParams.model_name;
        document.getElementById('accuracy-text').textContent = (modelParams.test_accuracy * 100).toFixed(2) + '%';
        console.log("Model parameters loaded successfully.");
    } catch (error) {
        console.error("Error loading model:", error);
        document.getElementById('model-name-text').textContent = "Error loading model";
    }
}

// Sigmoid function
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Prediction function
function predict(inputs) {
    if (!modelParams) return null;

    const { coefficients, intercept, scaler, feature_names } = modelParams;
    let z = intercept;

    // Scaling and Dot Product
    for (let i = 0; i < feature_names.length; i++) {
        const rawValue = inputs[feature_names[i]];
        const mean = scaler.mean[i];
        const scale = scaler.scale[i];
        
        // Scale the input: (x - mean) / scale
        const scaledValue = (rawValue - mean) / scale;
        
        // Add to z: W_i * X_i
        z += coefficients[i] * scaledValue;
    }

    const probability = sigmoid(z);
    return probability;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    loadModel();

    const themeToggle = document.getElementById('theme-toggle');
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

    // Form Submission
    const riskForm = document.getElementById('risk-form');
    riskForm.addEventListener('submit', (e) => {
        e.preventDefault();

        if (!modelParams) {
            alert("Model is still loading or failed to load. Please refresh.");
            return;
        }

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

        const prob = predict(inputs);
        displayResults(prob);
    });
});

function displayResults(prob) {
    const resultsSection = document.getElementById('results-section');
    const riskScore = document.getElementById('risk-score');
    const confidenceScore = document.getElementById('confidence-score');
    const riskAlert = document.getElementById('risk-alert');
    const alertTitle = document.getElementById('alert-title');
    const alertText = document.getElementById('alert-text');

    resultsSection.classList.remove('hidden');
    
    const percentage = (prob * 100).toFixed(1) + '%';
    riskScore.textContent = percentage;

    const isHighRisk = prob >= 0.5;
    const confidence = isHighRisk ? prob : (1 - prob);
    confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';

    if (isHighRisk) {
        riskAlert.className = 'risk-alert alert-high';
        alertTitle.textContent = '🚨 HIGH RISK ALERT';
        alertText.textContent = 'Diagnostic indicators suggest high cardiovascular risk. Immediate clinical consultation is prioritized.';
    } else {
        riskAlert.className = 'risk-alert alert-low';
        alertTitle.textContent = '✅ LOW RISK OBSERVED';
        alertText.textContent = 'Clinical metrics align with low risk profiles. Continue routine wellness and preventative measures.';
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

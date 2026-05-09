# HeartAssistX: Intelligent Patient Risk Assessment

> **From Predictive Analytics to Clinical Decision Support.**

HeartAssistX is an AI-driven clinical analytics system designed to bridge the gap between raw physiological data and actionable medical insights. It transitions traditional heart disease prediction into a modern, intelligent diagnostic assistant.

## The Problem
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection is critical, yet clinical datasets are often complex, making it difficult for practitioners to quickly quantify risk without sophisticated statistical tools. The challenge lies in identifying the subtle physiological drivers of cardiac risk while maintaining high diagnostic reliability.

## The Solution
HeartAssistX solves this by implementing a high-fidelity machine learning pipeline that processes UCI Heart Disease data to identify high-risk patients with precision. 

- **Intelligence**: Evolved from classical statistical modeling into an agentic interface that provides real-time risk probability and confidence scoring.
- **Precision**: Utilizes regularized Logistic Regression ($C=0.1$) to ensure the model generalizes well to new patient data, achieving a **88.52% test accuracy**.
- **Clinical Design**: Features a specialized dual-mode theme system (Clinical Red/White & High-Contrast Dark) tailored for low-latency clinical environments.

## Technology Stack
- **Engine**: Scikit-Learn (Logistic Regression, Decision Trees)
- **Logic**: Python (FastAPI, Pandas, NumPy)
- **Interface**: Streamlit & Static Web (HTML5/CSS3/JS)
- **Deployment**: Portable static web agent for serverless infrastructure.

## Key Performance
| Metric | Value |
| :--- | :--- |
| **Model Accuracy** | 88.52% |
| **Regularization** | L2 (C=0.1) |
| **Target Data** | UCI Heart Disease Repository |

---
**Lead Developer:** Shashwat Sinha  
**Core Team:** Shivam Mishra, Aaryan Yadav

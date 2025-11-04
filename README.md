# Healthcare Ensemble: Multi-Disease Risk Prediction API

**Predict diabetes, heart, and kidney disease risk using ensemble learning — production-ready, modular, and deployable.**
---

## Overview

This project combines **three medical datasets** (Diabetes, Heart Disease, Kidney Disease) into a **unified ensemble model** that:

- Predicts **likelihood of each disease**
- Generates **personalized health recommendations** using **Google Gemini**
- Runs as a **scalable FastAPI service**
- Is **fully containerized** with Docker
- Supports **CI/CD** via GitHub Actions

---

## Features

| Feature | Description |
|-------|-----------|
| **Multi-Disease Prediction** | Diabetes, Heart, Kidney in one model |
| **SMOTE + Logistic Regression** | Handles class imbalance |
| **Feature Alignment** | Combines heterogeneous datasets |
| **Gemini-Powered Insights** | AI-generated health advice |
| **Modular & Testable** | Clean architecture |

---

## Project Structure
healthcare_ensemble/
├── config/              # Settings & config
├── data/raw/            # Input CSV files
├── models/              # Trained models & scaler
├── src/
│   ├── preprocessing.py # Data cleaning & alignment
│   ├── models.py        # Training & ensemble logic
│   ├── inference.py     # Gemini recommendations
├── scripts/train.py     # Training pipeline
├── tests/               # Unit tests
├── requirements.txt

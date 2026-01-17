# AI-Powered Algorithmic Trader with NLP Risk Management 🛡️

A hybrid trading system that combines **XGBoost** for trend prediction with **VADER Sentiment Analysis** for risk control.

## 🚀 Key Features
* **Market Sentiment Engine:** Scrapes financial news and calculates a "Market Mood" score using NLP.
* **Automated Safety Switch:** Automatically disables trading during negative news cycles (Sentiment < -0.05).
* **Performance:**
    * **Defensive Mode:** successfully preserved 100% capital during simulated market crashes.
    * **Active Mode:** Experimental XGBoost model (Currently optimizing for higher precision).

## 🛠 Tech Stack
* **Core:** Python, Pandas, NumPy
* **ML:** XGBoost, Scikit-Learn
* **NLP:** VADER Sentiment
* **Automation:** SMTP Email Alerts

## 📉 Project Status
* **Risk Layer:** ✅ Stable (Production Ready)
* **Prediction Layer:** 🚧 In Development (Focusing on Feature Engineering)

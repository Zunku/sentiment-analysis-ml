# SentimentScope
## AI-Powered Sentiment Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)](https://python.org)
![SQL](https://img.shields.io/badge/SQL-black?style=for-the-badge&logo=sqlite&logoColor=white)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124.0+-00a393?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=101010)](https://fastapi.tiangolo.com)
[![Azure AI ML](https://img.shields.io/badge/Azure%20ML-1.7+-blue?style=for-the-badge&logo=airbrake&logoColor=white&labelColor=black)](https://azure.microsoft.com/es-mx)
[![Docker](https://img.shields.io/badge/Docker-Container-0078d7?style=for-the-badge&logo=docker&logoColor=white&labelColor=101010&logoSize=auto)](https://hub.docker.com)
[![Jupyter NB](https://img.shields.io/badge/Jupyter-notebooks-orange?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=101010&logoSize=auto)](https://jupyter.org)
![MLOps](https://img.shields.io/badge/MLOps-black?style=for-the-badge)

![](./assets/github-cover.png)

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#exploratory-data-analysis-eda">EDA</a> •
  <a href="#model-development">Model Development</a> •
  <a href="#api-development-with-fastapi">API & Docker</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#data-selection">Data Selection</a> •
  <a href="#how-to-run-locally">How to run</a>
</p>

<center> <b> SentimentScope </b> is an end-to-end sentiment analysis (NLP) project designed to demonstrate real-world machine learning engineering and MLOps skills. From exploratory data analysis to automated cloud retraining pipelines, this project showcases the full lifecycle of an ML system: exploration, modeling, deployment, monitoring, and scalability. </center>

## Key Features
- 📊 Complete Exploratory Data Analysis using Jupyter Notebooks
- 🖇️ Custom preprocessing and text vectorization pipeline
- 🤖 Multiple machine learning models evaluated and optimized
- ☁ Cloud-based training and automated retraining using Azure Machine Learning Pipelines
- 🌐 REST API built with FastAPI for real-time predictions
- 🐳 Dockerized deployment workflow for local and production environments

---

## Exploratory Data Analysis (EDA)
The analysis begins in Jupyter Notebooks, where the dataset is deeply explored to understand:
- **Insights** about E-commerce in Brazil such as:
  - Cities with the most orders.
  - Days and times of day with the most orders.
  - Evolution of e-commerce.
  - How money moves (shipping, average cost per order, state, category).
  - Shiping time.
- **Class distribution** and sentiment balance
- **Token frequency** and linguistic patterns
- Text length distributions
- Common **n-grams** for negative and positive classes
- Outliers and noise detection

Visualizations include histograms, lineplots, confusion matrix, and an interactive map with E-commerce distribution in Brazil

The **EDA** phase also includes cleaning procedures such as:
- Imputing missing values, parsing dates
- Removing HTML tags, special characters, and repeated symbols
- Lemmatization and stopword removal
- Handling of unbalanced classes

---

## Model Development
The modeling workflow is structured and reproducible:
- **TF-IDF** and other vectorization strategies tested
- Baseline models (Logistic Regression, Naive Bayes)
- **Hyperparameter tuning** using grid and randomized search
- **Cross-validation** and robust **metrics** (precision 0.8, recall 0.9, F1-Score 0.9)

The final model is serialized using pickle and stored along with the preprocessing pipeline.

---

## Azure Machine Learning: Cloud Training & Automated Pipelines
This project integrates with Azure Machine Learning to simulate **production-grade ML engineering.**

### Cloud Capabilities
- Remote compute clusters for **scalable model training**
- Dataset registration for versioned and reproducible data
- Environment dependencies tracked via conda specification
- Experiment tracking and model registry with MLFlow

### Automated Retraining Pipeline
An Azure ML Pipeline automates the entire training process, separated in diferent components:
1. Data ingestion and validation
2. Preprocessing and feature engineering
3. Model training and evaluation
4. Automatic registration of new models if performance improves

This allows **continuous improvement** of the model without manual intervention.

---

## API Development with FastAPI
A high-performance REST API exposes the model for **real-time** sentiment predictions.

### API Features
- POST endpoint for sending text and receiving predictions
- Built-in schema validation via Pydantic
- More under construction

The API integrates directly with the serialized preprocessing + modeling pipeline to ensure consistent predictions.

---

## Dockerized Deployment
The full system is prepared for containerized deployment.

### What Docker Enables
- Environment consistency across machines
- Fast local development and production alignment
- Easy orchestration via Docker Compose

The Docker image contains the FastAPI server, model artifacts, and all dependencies.

---

## Project Structure
```
project/
|── azure/                 # Azure ML config & Pipeline scripts
|   └── pipeline/
|       ├── src/           # Pipeline components
|       └── tmp/           # Temporal files
├── data/                  # Dataset handling
├── deployment/            # Dockerfile and deployment configs
│   └── app/               # FastAPI application
|       └── model/         # Model Inference Script
├── notebooks/             # Exploratory Data Analysis & Model Development
├── utils/                 # Custom modules
└── README.md
```

## Data Selection

This project uses the <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Brazilian E-Commerce Public Dataset by Olist</a> (Kaggle), a realistic and highly structured dataset that reflects real-world business data.

### Why This Dataset?
- Multiple **relational tables** (orders, customers, products, payments, reviews)
- Strong **SQL-style complexity** with joins, aggregations, and filters
- Real-world issues: missing values, noisy text, inconsistent categories
- Rich business context ideal for sentiment analysis

### How It Was Used
- Joined multiple tables to enrich customer reviews with transactional data
- Combined **text data** with **structured features**
- Simulated real production data modeling scenarios



---

## How to Run Locally
Need to download 
<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Brazilian E-Commerce Public Dataset by Olist</a> (Kaggle)

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Start the API
```
uvicorn deployment.app.main:app --reload
```

### 3. Build Docker Image
```
docker build -t sentiment-analysis .
```

### 4. Run Container
```
docker run -p 80:80 sentiment-analysis
```

---

## Future Improvements
- Creating an interfaz to interact with the API
- Batch inference system
- Full CI/CD integration with GitHub Actions

---

<p align="center">Made with passion for Machine Learning and scalable AI systems.</p>
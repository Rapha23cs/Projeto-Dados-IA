# 🧠 Churn Predictor SaaS

Um motor de Inteligência Artificial de ponta-a-ponta para prever e evitar o cancelamento (Churn) de clientes de telecomunicações.

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-110000?style=for-the-badge&logo=xgboost&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## 🚀 Arquitetura do Sistema

O projeto é dividido em um ambiente robusto de MLOps:
1. **Frontend (React + Vite)**: Interface "Glassmorphism" construída com CSS puro, rodando suave na GPU, com design voltado a produtos SaaS Premium.
2. **Backend (FastAPI)**: Microsserviço assíncrono para inferência de Machine Learning via API RESTful.
3. **Machine Learning (XGBoost)**: Modelo Sênior otimizado matematicamente, utilizando `scale_pos_weight` para lidar com desbalanceamento, *Feature Engineering* de Safra e Valor, e Threshold Tuning (Corte Ótimo via Curva PR).
4. **XAI (SHAP)**: Motor de Teoria dos Jogos acoplado ao FastAPI para não apenas entregar um Score de Risco, mas explicar matematicamente o *porquê* daquele risco.

## 🛠️ Como Executar

### 1. Iniciar o Backend (API)
```bash
# Na pasta raiz
python -m venv venv
.\venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```
> A API ficará disponível em: http://localhost:8000/docs

### 2. Iniciar o Frontend (UI)
```bash
# Em um novo terminal
cd frontend
npm install
npm run dev
```
> Acesse: http://localhost:5173

## 📊 Performance do Modelo (Fase 10)

- **Acurácia Estável**: ~78%
- **Recall da Classe de Risco (Churn)**: **75%** (Identifica 3 em cada 4 clientes em fuga).
- **Threshold Otimizado**: 57.56% (Maximiza o lucro do negócio, F1-Score).

## 🐋 Executando via Docker
O projeto conta com conteinerização pronta para uso em produção.
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

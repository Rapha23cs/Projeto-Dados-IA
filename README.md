# 🧠 Churn Predictor SaaS

Um motor de Inteligência Artificial de ponta-a-ponta para prever e evitar o cancelamento (Churn) de clientes de telecomunicações, construído com arquitetura Serverless/PaaS moderna.

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-110000?style=for-the-badge&logo=xgboost&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

## 🚀 Arquitetura do Sistema

O projeto é dividido em um ambiente moderno e desacoplado:
1. **Frontend (Vercel)**: Interface React + Vite "Glassmorphism" construída com CSS puro, rodando leve no navegador, com design voltado a produtos SaaS Premium.
2. **Backend (Render)**: Microsserviço assíncrono FastAPI para inferência de Machine Learning via API RESTful.
3. **Database (Supabase)**: Banco de dados PostgreSQL na nuvem para armazenar logs de predições e histórico de clientes.
4. **Machine Learning (XGBoost)**: Modelo Sênior otimizado matematicamente, utilizando `scale_pos_weight` para lidar com desbalanceamento, *Feature Engineering* de Safra e Valor, e Threshold Tuning (Corte Ótimo via Curva PR).
5. **XAI Direcional (Native XGBoost SHAP)**: Motor nativo de Teoria dos Jogos via `pred_contribs=True` acoplado à API, extraindo exatamente quais fatores reduzem (verde) ou aumentam (vermelho) o risco de cada cliente individual.

## 🛠️ Como Executar Localmente

### 1. Iniciar o Backend (API)
```bash
# Na pasta raiz
python -m venv venv
.\venv\Scripts\activate  # (Windows)
pip install -r requirements.txt

# Configure a URL do Supabase, se desejar salvar logs
# export SUPABASE_URL="sua-url"
# export SUPABASE_KEY="sua-key"

uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```
> A API ficará disponível em: http://localhost:8000/docs

### 2. Iniciar o Frontend (UI)
```bash
# Em um novo terminal
cd frontend
npm install

# Configure a URL da API
# Crie um arquivo .env local com: VITE_API_URL=http://localhost:8000

npm run dev
```
> Acesse: http://localhost:5173

## 📊 Performance do Modelo

- **Acurácia Estável**: ~78%
- **Recall da Classe de Risco (Churn)**: **75%** (Identifica 3 em cada 4 clientes em fuga).
- **Threshold Otimizado**: 57.56% (Maximiza o lucro do negócio, F1-Score).

## 🐋 Deploy (Docker)
O backend conta com conteinerização (Dockerfile single-stage otimizado para Python) pronto para deploy na nuvem.
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

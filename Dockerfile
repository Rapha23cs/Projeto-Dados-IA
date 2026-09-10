# Estágio 1: Build do Frontend (Node.js)
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
# Copiar arquivos de configuração do Node
COPY frontend/package*.json ./
RUN npm install
# Copiar o código fonte do frontend e buildar
COPY frontend/ ./
RUN npm run build

# Estágio 2: Backend e Produção (Python)
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instalar dependências do sistema necessárias para algumas bibliotecas de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas os requirements primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Instalar as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código fonte do backend para dentro do container
COPY . .

# Copiar o Frontend compilado (Estágio 1) para a pasta onde o FastAPI espera encontrar
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expor a porta 8000 para a API (FastAPI)
EXPOSE 8000

# Comando padrão ao rodar o container (Inicializar a API do FastAPI)
CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
